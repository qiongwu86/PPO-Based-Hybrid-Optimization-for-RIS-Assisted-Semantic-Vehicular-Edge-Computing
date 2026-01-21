import numpy as np
import copy
import math

# 导入环境配置
from environment import Environ, TASK_SIZE_MBIT

# ================= QPSO 配置参数 =================
QPSO_POP_SIZE = 50        # 粒子群规模
QPSO_ITERATIONS = 50      # 迭代次数
ALPHA_START = 1.0         # 收缩扩张系数初始值 (通常 1.0 或 0.8)
ALPHA_END = 0.5           # 收缩扩张系数结束值 (通常 0.5)

# 语义相似度约束参数
SEM_SIM_THRESHOLD = 0.9    # 语义相似度阈值
SEM_PENALTY_SCALE = 5.0    # 惩罚系数
USE_SEMANTIC_PENALTY = True

class QPSOSolver:
    def __init__(self, 
                 n_vehicles=None,
                 ris_row=None,
                 ris_col=None,
                 pop_size=None, 
                 iterations=None,
                 alpha_start=None,
                 alpha_end=None):
        """
        初始化 QPSO 求解器
        """
        # 使用配置参数或默认值
        self.pop_size = pop_size if pop_size is not None else QPSO_POP_SIZE
        self.iterations = iterations if iterations is not None else QPSO_ITERATIONS
        self.alpha_start = alpha_start if alpha_start is not None else ALPHA_START
        self.alpha_end = alpha_end if alpha_end is not None else ALPHA_END
        
        # 存储配置
        self.n_vehicles = n_vehicles
        self.ris_row = ris_row
        self.ris_col = ris_col
        self.n_ris_elements = None
        
        # 语义约束
        self.sem_sim_threshold = SEM_SIM_THRESHOLD
        self.sem_penalty_scale = SEM_PENALTY_SCALE
        self.use_semantic_penalty = USE_SEMANTIC_PENALTY
        
        # 维度索引
        self.idx_ris_end = None
        self.idx_k_end = None
        self.idx_lam_end = None
        self.dim = None

        # 粒子群状态
        self.particles = None       # 当前位置 X
        self.p_best = None          # 个体历史最优 Pbest
        self.p_best_fitness = None  # 个体历史最优适应度
        self.g_best = None          # 全局最优 Gbest
        self.g_best_fitness = -float('inf')

    def _infer_config_from_env(self, env):
        """从环境对象推断配置"""
        if self.n_vehicles is None:
            self.n_vehicles = env.n_total
        if self.ris_row is None:
            self.ris_row = env.ris.M
        if self.ris_col is None:
            self.ris_col = env.ris.N
        
        self.n_ris_elements = self.ris_row * self.ris_col
        
        # 计算维度索引
        self.idx_ris_end = self.n_ris_elements
        self.idx_k_end = self.n_ris_elements + self.n_vehicles
        self.idx_lam_end = self.n_ris_elements + self.n_vehicles + (3 * self.n_vehicles)
        self.dim = self.idx_lam_end

    def init_particles(self):
        """初始化粒子群"""
        if self.dim is None:
            raise ValueError("Configuration not initialized.")
        
        # 初始化位置矩阵 (Pop_Size, Dim)
        self.particles = np.zeros((self.pop_size, self.dim))
        
        # 1. RIS 相位 [0, 2pi]
        self.particles[:, :self.idx_ris_end] = np.random.uniform(0, 2*np.pi, (self.pop_size, self.n_ris_elements))
        
        # 2. K值 [1, 20]
        self.particles[:, self.idx_ris_end:self.idx_k_end] = np.random.randint(1, 21, (self.pop_size, self.n_vehicles))
        
        # 3. Lambda 权重 [0, 1]
        self.particles[:, self.idx_k_end:] = np.random.uniform(0, 1, (self.pop_size, 3 * self.n_vehicles))
        
        # 初始化 Pbest 为当前位置
        self.p_best = self.particles.copy()
        self.p_best_fitness = np.full(self.pop_size, -float('inf'))
        
        # Gbest 稍后计算

    def decode_particle(self, particle):
        """解码单个粒子为物理参数"""
        # A. RIS 相位
        ris_phases = particle[:self.idx_ris_end]
        
        # B. K 值 (取整)
        k_values = np.round(particle[self.idx_ris_end:self.idx_k_end]).astype(int)
        k_values = np.clip(k_values, 1, 20)
        
        # C. Lambda 比例 (归一化)
        raw_lam = particle[self.idx_k_end:].reshape(self.n_vehicles, 3)
        sum_lam = np.sum(raw_lam, axis=1, keepdims=True) + 1e-9
        lambdas = raw_lam / sum_lam 
        
        return ris_phases, k_values, lambdas

    def evaluate_fitness(self, population, env_snapshot, return_details=False):
        """
        计算适应度 (与 GA 逻辑基本一致)
        目标: Maximize (-MinMaxDelay)
        """
        if self.dim is None:
            self._infer_config_from_env(env_snapshot)
            
        fitness_scores = []
        delay_details = [] if return_details else None
        actual_pop_size = population.shape[0]
        
        # 备份环境状态
        original_phases = env_snapshot.ris.phases.copy()
        task_bits = TASK_SIZE_MBIT * 1e6
        
        for i in range(actual_pop_size):
            ind = population[i]
            ris_phases, k_values, lambdas = self.decode_particle(ind)
            
            # 设置环境
            env_snapshot.ris.set_phases(ris_phases)
            
            max_delays = []
            semantic_penalty = 0.0
            
            for v_idx in range(self.n_vehicles):
                k_val = k_values[v_idx]
                lam_loc, lam_edge, lam_bs = lambdas[v_idx]
                
                srv_idx, _ = env_snapshot.find_nearest_service_vehicle(v_idx)
                
                snr_v2v = 1e-10
                if srv_idx != -1:
                    snr_v2v = env_snapshot.get_v2v_sinr(v_idx, srv_idx)
                
                snr_v2i = env_snapshot.get_v2i_sinr(v_idx)
                
                A_l, A_e, A_b = env_snapshot.calculate_delay_coefficients(
                    task_bits, snr_v2v, snr_v2i, k_val
                )
                
                d_loc = A_l * lam_loc
                d_edge = A_e * lam_edge
                d_bs = A_b * lam_bs
                
                veh_total_delay = max(d_loc, d_edge, d_bs)
                max_delays.append(veh_total_delay)
                
                # 语义惩罚
                if self.use_semantic_penalty:
                    delta_v2i = env_snapshot.sem_model.get_similarity(snr_v2i, k_val)
                    if delta_v2i < self.sem_sim_threshold:
                        penalty = self.sem_penalty_scale * (self.sem_sim_threshold - delta_v2i)
                        semantic_penalty += penalty
            
            system_objective = np.max(max_delays)
            total_objective = system_objective + semantic_penalty
            
            # 适应度 = 负的时延 (越大越好)
            fitness_scores.append(-total_objective)
            
            if return_details:
                delay_details.append({
                    'real_delay': system_objective,
                    'penalty': semantic_penalty,
                    'total_objective': total_objective
                })
        
        env_snapshot.ris.set_phases(original_phases)
        
        if return_details:
            return np.array(fitness_scores), delay_details
        return np.array(fitness_scores)

    def solve(self, env_snapshot, verbose=False):
        """
        QPSO 主求解循环
        """
        if self.dim is None:
            self._infer_config_from_env(env_snapshot)
            
        self.init_particles()
        
        # 初始评估
        fitness = self.evaluate_fitness(self.particles, env_snapshot)
        self.p_best = self.particles.copy()
        self.p_best_fitness = fitness.copy()
        
        best_idx = np.argmax(fitness)
        self.g_best = self.particles[best_idx].copy()
        self.g_best_fitness = fitness[best_idx]
        
        best_real_delay = None
        best_penalty = None
        
        # QPSO 迭代
        for t in range(self.iterations):
            # 1. 计算 Alpha (线性衰减)
            alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * (t / self.iterations)
            
            # 2. 计算平均最好位置 (Mean Best Position, mbest)
            # mbest = mean(P_best)
            mbest = np.mean(self.p_best, axis=0)
            
            # 3. 向量化更新粒子位置
            # Phi ~ U(0, 1)
            phi = np.random.rand(self.pop_size, self.dim)
            
            # 本地吸引子: p_{i,d} = phi * pbest + (1-phi) * gbest
            p_attractor = phi * self.p_best + (1 - phi) * self.g_best
            
            # u ~ U(0, 1)
            u = np.random.rand(self.pop_size, self.dim)
            
            # 随机正负号 (50% 概率)
            sign = np.where(np.random.rand(self.pop_size, self.dim) > 0.5, 1, -1)
            
            # QPSO 位置更新公式:
            # X(t+1) = P +/- alpha * |mbest - X(t)| * ln(1/u)
            dist_to_mbest = np.abs(mbest - self.particles)
            self.particles = p_attractor + sign * alpha * dist_to_mbest * np.log(1 / u)
            
            # 4. 边界约束处理
            # RIS 相位: Modulo 2pi
            self.particles[:, :self.idx_ris_end] = np.mod(self.particles[:, :self.idx_ris_end], 2*np.pi)
            # K 值: Clip [1, 20]
            self.particles[:, self.idx_ris_end:self.idx_k_end] = np.clip(
                self.particles[:, self.idx_ris_end:self.idx_k_end], 1, 20
            )
            # Lambda: Clip [0.01, 1] (解码时会再次归一化，这里clip防止负数)
            self.particles[:, self.idx_k_end:] = np.clip(
                self.particles[:, self.idx_k_end:], 0.01, 1.0
            )
            
            # 5. 评估新位置
            if verbose and t % 10 == 0:
                fitness, details = self.evaluate_fitness(self.particles, env_snapshot, return_details=True)
            else:
                fitness = self.evaluate_fitness(self.particles, env_snapshot, return_details=False)
                details = None
            
            # 6. 更新 Pbest 和 Gbest
            # 找出本轮优于历史最优的粒子索引
            improved_indices = fitness > self.p_best_fitness
            
            # 更新这些粒子的 Pbest
            self.p_best[improved_indices] = self.particles[improved_indices]
            self.p_best_fitness[improved_indices] = fitness[improved_indices]
            
            # 更新 Gbest
            current_best_idx = np.argmax(self.p_best_fitness)
            if self.p_best_fitness[current_best_idx] > self.g_best_fitness:
                self.g_best = self.p_best[current_best_idx].copy()
                self.g_best_fitness = self.p_best_fitness[current_best_idx]
                
                # 如果有详细信息，记录下来
                if details is not None:
                    # 注意：details 对应的是当前 particles 的 fitness
                    # 如果 gbest 是这一代产生的，我们可以直接取
                    # 如果 gbest 是以前产生的，这里可能会有一点误差，但在 verbose 打印中影响不大
                    # 为了精确，我们最后会重新解码一遍
                    pass

            if verbose and t % 10 == 0:
                print(f"QPSO Iter {t}: Best Objective = {-self.g_best_fitness:.4f} s")

        # 最终解码
        best_ris, best_k, best_lam = self.decode_particle(self.g_best)
        
        # 计算最终真实指标
        _, final_details = self.evaluate_fitness(np.array([self.g_best]), env_snapshot, return_details=True)
        best_real_delay = final_details[0]['real_delay']
        best_penalty = final_details[0]['penalty']
        
        return best_ris, best_k, best_lam, best_real_delay, best_penalty, -self.g_best_fitness

# ================= 测试代码 =================
if __name__ == "__main__":
    TEST_N_VEHICLES = 15
    TEST_RIS_ROW = 6
    TEST_RIS_COL = 6
    
    # 1. 创建环境
    env = Environ(n_vehicles=TEST_N_VEHICLES, dt=0.1)
    
    # 手动同步 RIS 设置
    if TEST_RIS_ROW != env.ris.M or TEST_RIS_COL != env.ris.N:
        env.ris.M = TEST_RIS_ROW
        env.ris.N = TEST_RIS_COL
        env.ris.num_elements = TEST_RIS_ROW * TEST_RIS_COL
        env.ris.element_positions = np.zeros((env.ris.num_elements, 3))
        env.ris.phases = np.zeros(env.ris.num_elements)
        env.ris._init_geometry_yz()
        env._compute_static_channel()
    
    # 2. 初始化 QPSO
    qpso = QPSOSolver(pop_size=50, iterations=50)
    
    print("="*60)
    print("Running QPSO Optimization...")
    print(f"Configuration: {TEST_N_VEHICLES} vehicles")
    print("="*60)
    
    # 3. 求解
    ris, k, lam, delay, pen, obj = qpso.solve(env, verbose=True)
    
    print("\n" + "="*60)
    print("=== QPSO Result ===")
    print(f"Real Delay:       {delay:.4f} s")
    print(f"Penalty:          {pen:.4f}")
    print(f"Total Objective:  {obj:.4f}")
    print(f"Sample K:         {k[:5]}")
    print(f"Sample Lambda:    {lam[0]}")
    print("="*60)