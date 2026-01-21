import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np
import math

# 导入环境定义
from environment import Environ, ROAD_LENGTH, RIS_LOC, BS_LOC, POISSON_RATE, TASK_SIZE_MBIT

# ================= 场景配置 (固定维度) =================
N_VEHICLES = 15          # 默认车辆数目
RIS_ROW = 5
RIS_COL = 5
N_RIS_ELEMENTS = RIS_ROW * RIS_COL

# 状态维度计算
# 1. RIS相位(36) + 2. 位置(10*2) + 3. AoA(10*2) + 4. SINR(10)
STATE_DIM = N_RIS_ELEMENTS + (N_VEHICLES * 2) + (N_VEHICLES * 2) + N_VEHICLES
# 动作维度计算
# RIS相位(36) + k值(10)
ACTION_DIM = N_RIS_ELEMENTS + N_VEHICLES

print(f"--- Dimension Check ---")
print(f"Vehicles: {N_VEHICLES}, RIS Elements: {N_RIS_ELEMENTS}")
print(f"State Dim: {STATE_DIM}")
print(f"Action Dim: {ACTION_DIM}")
print(f"-----------------------")

# PPO 超参数
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
GAMMA = 0.6
K_EPOCHS = 4
EPS_CLIP = 0.2
ACTION_STD = 0.5

# 语义相似度约束
SEM_SIM_THRESHOLD = 0.9
SEM_PENALTY_SCALE = 5.0  # 惩罚系数（越大约束越强）

MAX_EPISODES = 5000
MAX_STEPS = 50
UPDATE_TIMESTEP = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 辅助函数：计算维度 =================
def calculate_dims(n_vehicles, ris_row, ris_col):
    """计算给定配置下的状态和动作维度"""
    n_ris_elements = ris_row * ris_col
    state_dim = n_ris_elements + (n_vehicles * 2) + (n_vehicles * 2) + n_vehicles
    action_dim = n_ris_elements + n_vehicles
    return state_dim, action_dim, n_ris_elements

# ================= 1. 环境包装器 (定制状态) =================

class SpecificStateEnv:
    def __init__(self, n_vehicles=None, ris_row=None, ris_col=None):
        # 支持参数化配置，如果没有提供则使用全局默认值
        self.n_vehicles = n_vehicles if n_vehicles is not None else N_VEHICLES
        self.ris_row = ris_row if ris_row is not None else RIS_ROW
        self.ris_col = ris_col if ris_col is not None else RIS_COL
        self.n_ris_elements = self.ris_row * self.ris_col
        
        # 初始化环境，仅需总车辆数
        self.env = Environ(n_vehicles=self.n_vehicles, dt=0.1)
        
        # 强制修改环境内部的 RIS 尺寸
        self.env.ris.M = self.ris_row
        self.env.ris.N = self.ris_col
        self.env.ris.num_elements = self.n_ris_elements
        # 重新生成几何位置
        self.env.ris.element_positions = np.zeros((self.n_ris_elements, 3))
        self.env.ris.phases = np.zeros(self.n_ris_elements)
        self.env.ris._init_geometry_yz()
        # 重新计算静态信道
        self.env._compute_static_channel()
        
        # 记录当前的 RIS 相位作为状态的一部分 (初始化为0)
        self.current_ris_phases = np.zeros(self.n_ris_elements)

    def reset(self):
        self.env = Environ(n_vehicles=self.n_vehicles, dt=0.1)
        # 同样需要覆盖 RIS 设置
        self.env.ris.M = self.ris_row
        self.env.ris.N = self.ris_col
        self.env.ris.num_elements = self.n_ris_elements
        # 重新分配数组大小以匹配新的RIS尺寸
        self.env.ris.element_positions = np.zeros((self.n_ris_elements, 3))
        self.env.ris.phases = np.zeros(self.n_ris_elements)
        self.env.ris._init_geometry_yz()
        self.env._compute_static_channel()
        
        self.current_ris_phases = np.zeros(self.n_ris_elements)
        return self._get_state()

    def _calculate_aoa(self, ris_pos, veh_pos):
        """
        计算车辆相对于 RIS 中心的到达角 (Angle of Arrival)
        返回: [Azimuth, Elevation] (弧度)
        """
        # 向量：车辆 -> RIS
        vec = veh_pos - ris_pos
        x, y, z = vec[0], vec[1], vec[2]
        dist = np.linalg.norm(vec) + 1e-9
        
        # 方位角 (Azimuth): 在 XY 平面的投影角度
        azimuth = np.arctan2(y, x) 
        
        # 仰角 (Elevation): Z轴与平面的夹角
        elevation = np.arcsin(z / dist)
        
        return azimuth, elevation

    def _get_state(self):
        """
        构建状态向量 S_t = [RIS_Phases, Veh_Pos, Veh_AoA, V2I_SINR]
        维度: 36 + 20 + 20 + 10 = 86
        """
        # 必须先更新信道以获取 SINR
        self.env._update_v2i_dynamic_channels()
        
        # 1. 当前 RIS 相位 (36) - 归一化到 [-1, 1] (相位本身是 0~2pi)
        # 假设 self.current_ris_phases 存储的是弧度
        state_phases = np.cos(self.current_ris_phases) # 使用 cos 值可能比线性归一化更好，或者直接归一化
        # 这里为了简单，直接线性映射: [0, 2pi] -> [-1, 1]
        state_phases = (self.current_ris_phases / np.pi) - 1.0
        
        # 收集车辆数据
        pos_list = []
        aoa_list = []
        sinr_list = []
        
        for i in range(self.n_vehicles):
            veh = self.env.vehicles[i]
            
            # 2. 车辆位置 (2) - 归一化
            norm_x = veh.position[0] / 7.0
            norm_y = veh.position[1] / ROAD_LENGTH
            pos_list.extend([norm_x, norm_y])
            
            # 3. 到达角 (2) - 归一化
            # 输入范围大致是 [-pi, pi] -> [-1, 1]
            az, el = self._calculate_aoa(RIS_LOC, veh.position)
            aoa_list.extend([az / np.pi, el / (np.pi/2)]) # Elevation通常在 -pi/2 到 pi/2
            
            # 4. V2I SINR (1) - 归一化
            # 获取 SINR (线性值) -> dB
            sinr_lin = self.env.get_v2i_sinr(i)
            sinr_db = 10 * np.log10(sinr_lin + 1e-15)
            # 假设 SINR 范围 -20dB 到 30dB -> 映射到 -1 到 1
            # (val - min) / (max - min) * 2 - 1
            norm_sinr = (sinr_db - 5.0) / 25.0 # 中心化处理
            sinr_list.append(norm_sinr)
            
        # 拼接所有状态
        state = np.concatenate([
            state_phases,           # 36
            np.array(pos_list),     # 20
            np.array(aoa_list),     # 20
            np.array(sinr_list)     # 10
        ])
        
        return state.astype(np.float32)

    def step(self, action):
        # --- 1. 动作解析 ---
        # RIS部分
        ris_act = action[:self.n_ris_elements]
        # Action [-1, 1] -> Phase [0, 2pi]
        ris_phases = (ris_act + 1) * np.pi 
        self.env.ris.set_phases(ris_phases)
        self.current_ris_phases = ris_phases # 更新状态记录
        
        # k值部分
        k_act = action[self.n_ris_elements:]
        # Action [-1, 1] -> k [1, 20]
        k_values = np.round((k_act + 1) / 2 * 19 + 1).astype(int)
        k_values = np.clip(k_values, 1, 20)
        
        # --- 2. 物理演进 ---
        self.env.step_movement()
        
        # --- 3. 计算奖励 (基于新的 RIS 相位和 k 值) ---
        step_delays = []
        sem_penalties = []
        
        for i in range(self.n_vehicles):
            # 强制假设每辆车都有任务进行评估
            k_val = k_values[i]
            
            # 匹配服务车
            srv_idx, _ = self.env.find_nearest_service_vehicle(i)
            
            # 获取物理指标
            snr_v2v = 1e-10
            if srv_idx != -1:
                snr_v2v = self.env.get_v2v_sinr(i, srv_idx)
            snr_v2i = self.env.get_v2i_sinr(i)
            
            # 计算时延系数
            task_bits = TASK_SIZE_MBIT * 1e6
            A_l, A_e, A_b = self.env.calculate_delay_coefficients(task_bits, snr_v2v, snr_v2i, k_val)
            
            # LP 求解
            try:
                from lambda_lp_solver import compute_optimal_lambda
                _, _, _, t_opt = compute_optimal_lambda(A_l, A_e, A_b, prefer="closed_form")
            except:
                t_opt = max(A_l, A_e, A_b) # Fallback
            
            step_delays.append(t_opt)

            # 语义相似度惩罚（针对 V2I 链路）
            delta_v2i = self.env.sem_model.get_similarity(snr_v2i, k_val)
            if delta_v2i < SEM_SIM_THRESHOLD:
                sem_penalties.append(SEM_PENALTY_SCALE * (SEM_SIM_THRESHOLD - delta_v2i))
            
        # 奖励计算
        if len(step_delays) > 0:
            max_delay = np.max(step_delays)
            avg_delay = np.mean(step_delays)
            # 目标：最小化时延，并惩罚语义相似度不足
            reward = - (max_delay * 0.7 + avg_delay * 0.3) * 10
            if max_delay > 1.0:
                reward -= 2.0  # 时延硬惩罚
            if len(sem_penalties) > 0:
                reward -= float(np.sum(sem_penalties))
        else:
            reward = 0.0
            
        next_state = self._get_state()
        done = False
        
        return next_state, reward, done, {}

# ================= 2. PPO 算法 (Standard) =================

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # 共享层: 提取物理特征
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state):
        features = self.shared_net(state)
        action_mean = self.actor(features)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = Normal(action_mean, torch.sqrt(self.action_var))
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(features)
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        features = self.shared_net(state)
        action_mean = self.actor(features)
        action_var = self.action_var.expand_as(action_mean)
        dist = Normal(action_mean, torch.sqrt(action_var))
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim):
        self.action_std = ACTION_STD
        self.gamma = GAMMA
        self.eps_clip = EPS_CLIP
        self.K_epochs = K_EPOCHS
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, self.action_std).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': LR_ACTOR},
            {'params': self.policy.critic.parameters(), 'lr': LR_CRITIC}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, self.action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.cpu().numpy().flatten()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # 将多维动作的对数概率与熵按维度求和，得到每个样本的标量
            logprobs = logprobs.sum(dim=1)
            old_logprobs_sum = old_logprobs.sum(dim=1)
            dist_entropy = dist_entropy.sum(dim=1)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs_sum.detach())
            advantages = rewards - old_state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        
    def decay_action_std(self, rate, min_std):
        self.action_std = max(self.action_std - rate, min_std)
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)

class RolloutBuffer:
    def __init__(self):
        self.actions = []; self.states = []; self.logprobs = []
        self.rewards = []; self.state_values = []; self.is_terminals = []
    def clear(self):
        del self.actions[:]; del self.states[:]; del self.logprobs[:]
        del self.rewards[:]; del self.state_values[:]; del self.is_terminals[:]

# ================= 3. 训练入口 =================

def train():
    env = SpecificStateEnv()
    ppo_agent = PPO(STATE_DIM, ACTION_DIM)
    
    time_step = 0
    i_episode = 0
    reward_history = []
    
    while i_episode < MAX_EPISODES:
        state = env.reset()
        current_ep_reward = 0

        for t in range(MAX_STEPS):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward

            if time_step % UPDATE_TIMESTEP == 0:
                ppo_agent.update()
                
        i_episode += 1
        if i_episode % 100 == 0:
            ppo_agent.decay_action_std(0.05, 0.1)

        avg_reward = current_ep_reward / MAX_STEPS
        reward_history.append(avg_reward)
        if i_episode % 10 == 0:
            print(f"Episode: {i_episode} \t Avg Reward: {avg_reward:.4f} \t Std: {ppo_agent.action_std:.2f}")

    print("Training Finished.")
    torch.save(ppo_agent.policy.state_dict(), 'ppo_specific_veh15_RIS7x7.pth')

    # 绘制并保存奖励曲线
    if len(reward_history) > 0:
        plt.figure()
        plt.plot(np.arange(len(reward_history)) + 1, reward_history, label='Avg reward')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('PPO Training Reward')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('reward_curve_veh15_RIS7x7.png', dpi=200)
        plt.close()

    # 也保存原始奖励数据
    np.save('reward_history_veh15_RIS7x7.npy', np.array(reward_history, dtype=np.float32))

if __name__ == '__main__':
    train()