import numpy as np
import scipy.io
from scipy.interpolate import RegularGridInterpolator
from scipy.special import j0
import os
import math

# 尝试导入优化求解器
try:
    from lambda_lp_solver import compute_optimal_lambda
    HAS_SOLVER = True
except ImportError:
    print("[Warning] 'lambda_lp_solver.py' not found. Using static splitting.")
    HAS_SOLVER = False

# ================= 配置参数 =================
# 1. 几何与物理场景
ROAD_LENGTH = 300.0
LANE_CENTERS = [1.75, 5.25]
BS_LOC = np.array([-10.0, 150.0, 25.0])
RIS_LOC = np.array([10.0, 175.0, 25.0])

# 2. 通信参数
FC = 3e9
LIGHT_SPEED = 3e8
LAMBDA = LIGHT_SPEED / FC
BANDWIDTH = 360e3  # 360 kHz (每个链路的带宽)
NUM_BS_ANTENNAS = 8

# 噪声与功率
# 噪声功率谱密度：热噪声 -174 dBm/Hz = -204 dBW/Hz
NOISE_PSD_DBM_PER_HZ = -174  # dBm/Hz
NOISE_PSD_DBW_PER_HZ = NOISE_PSD_DBM_PER_HZ - 30  # dBW/Hz
NOISE_PSD_W_PER_HZ = 10 ** (NOISE_PSD_DBW_PER_HZ / 10.0)  # W/Hz
# 噪声功率 = 噪声功率谱密度 * 带宽
NOISE_POWER_BS = NOISE_PSD_W_PER_HZ * BANDWIDTH  # Watts (基站接收机)
NOISE_POWER_BS_DBM = 10 * np.log10(NOISE_POWER_BS * 1000)  # dBm (用于参考)
# V2V接收机噪声功率
NOISE_POWER_V2V = NOISE_PSD_W_PER_HZ * BANDWIDTH  # Watts (V2V接收机)
TX_POWER_VEHICLE = 0.2  # Watts

# 3. 增益与衰落
G_BS_LIN = 10 ** (8 / 10.0)
G_VEH_TX_LIN = 10 ** (3 / 10.0)
G_VEH_RX_LIN = 10 ** (3 / 10.0)
G_RIS_ELE_LIN = 1.0 

ALPHA_DIRECT = 3.5  # V2I NLoS
ALPHA_RIS = 2.2     # RIS LoS
ALPHA_V2V = 2.2     # V2V LoS
SHADOW_STD = 8.0
DECORR_DIST = 10.0

# 4. RIS 参数
RIS_ROWS = 6
RIS_COLS = 6
RIS_SPACING = LAMBDA / 2

# 5. 计算任务参数 (新增)
CPU_FREQ_LOCAL = 2e9    # 1 GHz (任务车辆)
CPU_FREQ_SERVICE = 2e9  # 3 GHz (服务车辆)
CPU_FREQ_BS = 6e9       # 6 GHz (基站)
CYCLES_PER_BIT = 1000   # 1 bit 需要 1000 cycles

POISSON_RATE = 4.0      # 任务到达率 (tasks/sec)
TASK_SIZE_MBIT = 0.4    # 平均任务大小 (Mbit)

# 6. 语义通信参数
SEMANTIC_K_DEFAULT = 4  # 默认语义符号数
I_AVG = 100.0           # suts/sentence
AH = 1200               # bits/sentence (硬件开销)
AW = 20                 # words/sentence

# 7. 车辆速度
VEH_VELOCITY_MEAN = 20.0 # m/s


# ================= 类定义 =================

class SemanticModel:
    """处理语义相似度查找表的类"""
    def __init__(self, filepath="sem_table.mat"):
        self.interpolator = None
        self.snr_grid = np.arange(-10, 21, 1)
        self.k_grid = np.arange(1, 21, 1)

        try:
            mat_data = scipy.io.loadmat(filepath)
            # 自动寻找变量名
            key = next((k for k in mat_data.keys() if not k.startswith('__')), None)
            if key:
                self.sim_matrix = mat_data[key]
                expected_shape = (len(self.snr_grid), len(self.k_grid))
                if self.sim_matrix.shape != expected_shape:
                    if self.sim_matrix.shape == (expected_shape[1], expected_shape[0]):
                        self.sim_matrix = self.sim_matrix.T
                
                self.interpolator = RegularGridInterpolator(
                    (self.snr_grid, self.k_grid), self.sim_matrix, bounds_error=False, fill_value=None
                )
        except Exception:
            pass

    def get_similarity(self, snr_linear, k_val):
        if self.interpolator is None: return 0.5 
        safe_snr = max(snr_linear, 1e-10)
        snr_db = 10 * np.log10(safe_snr)
        return float(self.interpolator((np.clip(snr_db, -10.0, 20.0), np.clip(k_val, 1, 20))))


class Task:
    """任务对象"""
    def __init__(self, id, source_veh_id, size_mbit, arrival_time):
        self.id = id
        self.source_veh_id = source_veh_id
        self.size_mbit = size_mbit
        self.size_bits = size_mbit * 1e6
        self.arrival_time = arrival_time
        
        # 优化结果
        self.lambda_local = 1.0
        self.lambda_edge = 0.0
        self.lambda_bs = 0.0
        self.total_delay = 0.0


class Vehicle:
    def __init__(self, id, start_y, lane_idx, velocity, veh_type='task'):
        self.id = id
        self.veh_type = veh_type # 'task' or 'service'
        self.lane_idx = lane_idx
        self.velocity = velocity
        # Position: [X, Y, Z]
        self.position = np.array([LANE_CENTERS[lane_idx], start_y, 0.0])
        
        # Channel States
        self.shadowing_db = np.random.normal(0, SHADOW_STD)
        self.h_small = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2)

    def update_position(self, delta_t):
        """
        更新车辆位置，考虑边界情况（超出边界时从另一端进入）
        使用模运算确保位置始终在[0, ROAD_LENGTH)范围内
        当跨越边界时变换车道（只发生在边界情况）
        """
        dist_moved = self.velocity * delta_t
        old_position_y = self.position[1]
        self.position[1] += dist_moved
        
        reset_flag = False
        lane_changed = False
        
        # 处理边界：使用模运算，超出边界时从另一端进入
        # Python的模运算会自动处理负数情况（-10 % 300 = 290）
        if self.position[1] >= ROAD_LENGTH or self.position[1] < 0:
            # 判断是否跨越边界（从一端到另一端）
            crossed_boundary = False
            if old_position_y >= 0 and old_position_y < ROAD_LENGTH:
                # 之前位置在有效范围内，现在超出边界
                crossed_boundary = True
            
            self.position[1] = self.position[1] % ROAD_LENGTH
            self.shadowing_db = np.random.normal(0, SHADOW_STD)
            reset_flag = True
            
            # 边界情况下：跨越边界时变换车道
            # 从车道0换到车道1，或从车道1换到车道0
            if crossed_boundary:
                self.lane_idx = 1 - self.lane_idx  # 切换车道（0<->1）
                # 更新X坐标以匹配新车道
                self.position[0] = LANE_CENTERS[self.lane_idx]
                lane_changed = True
        
        return dist_moved, reset_flag, lane_changed

    def update_v2i_fading(self, delta_t, dist_moved):
        rho_shadow = np.exp(-dist_moved / DECORR_DIST)
        self.shadowing_db = rho_shadow * self.shadowing_db + \
                            np.sqrt(1 - rho_shadow ** 2) * np.random.normal(0, SHADOW_STD)
        f_d = self.velocity / LAMBDA
        rho_small = j0(2 * np.pi * f_d * delta_t)
        n_complex = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2)
        self.h_small = rho_small * self.h_small + np.sqrt(1 - rho_small ** 2) * n_complex


class PassiveRIS:
    """固定位置的被动 RIS (YZ 平面)"""
    def __init__(self, M, N, loc, spacing):
        self.M = M
        self.N = N
        self.loc = loc
        self.spacing = spacing
        self.num_elements = M * N
        self.element_positions = np.zeros((self.num_elements, 3))
        self.phases = np.zeros(self.num_elements)
        self._init_geometry_yz()

    def _init_geometry_yz(self):
        idx = 0
        width_y = (self.N - 1) * self.spacing
        height_z = (self.M - 1) * self.spacing
        fixed_x = self.loc[0]
        start_y = self.loc[1] - width_y / 2
        start_z = self.loc[2] + height_z / 2 
        
        for m in range(self.M):
            z = start_z - m * self.spacing
            for n in range(self.N):
                y = start_y + n * self.spacing
                self.element_positions[idx] = [fixed_x, y, z]
                idx += 1

    def set_phases(self, phase_vector):
        self.phases = phase_vector

    def get_reflection_matrix(self):
        return np.exp(1j * self.phases)


class Environ:
    def __init__(self, n_vehicles=10, dt=0.1):
        # 仅保留总车辆数，不再区分任务车/服务车
        self.n_task_veh = n_vehicles  # 兼容后续任务遍历
        self.n_total = n_vehicles
        self.dt = dt
        self.current_time = 0
        
        self.sem_model = SemanticModel("sem_table.mat")
        
        # 初始化车辆
        self.vehicles = []
        # 所有车辆同质，不再区分任务/服务
        for i in range(n_vehicles):
            y_start = np.random.uniform(0, ROAD_LENGTH)
            lane = np.random.randint(0, 2)
            vel = np.random.normal(VEH_VELOCITY_MEAN, 2.0)
            self.vehicles.append(Vehicle(i, y_start, lane, vel, 'veh'))
        
        # 初始化 RIS
        self.ris = PassiveRIS(RIS_ROWS, RIS_COLS, RIS_LOC, RIS_SPACING)
        self.bs_ant_indices = np.arange(NUM_BS_ANTENNAS)
        
        # 物理层信道缓存
        self._H_rd_static = None 
        self._cached_h_sr_list = None
        self._cached_h_d_list = None
        
        # V2V 状态矩阵 (N_total x N_total)
        self.v2v_shadowing_db = np.random.normal(0, SHADOW_STD, (self.n_total, self.n_total))
        self.v2v_shadowing_db = (self.v2v_shadowing_db + self.v2v_shadowing_db.T) / 2
        self.v2v_h_small = (np.random.normal(0, 1, (self.n_total, self.n_total)) + 
                            1j * np.random.normal(0, 1, (self.n_total, self.n_total))) / np.sqrt(2)
        self.v2v_h_small = (self.v2v_h_small + self.v2v_h_small.T) / 2
        np.fill_diagonal(self.v2v_h_small, 0)

        self.task_counter = 0
        self._compute_static_channel()

    # --- 物理层辅助函数 ---
    def _get_bs_steering_vector(self, angle_elevation):
        psi = 2 * np.pi * 0.5 * np.sin(angle_elevation)
        response = np.exp(1j * self.bs_ant_indices * psi) / np.sqrt(NUM_BS_ANTENNAS)
        return response.reshape(-1, 1)

    def _compute_static_channel(self):
        """计算 H_rd (RIS -> BS)"""
        MN = self.ris.num_elements
        ris_pos = self.ris.element_positions
        
        H_rd = np.zeros((NUM_BS_ANTENNAS, MN), dtype=complex)
        vec_bs_ris = ris_pos - BS_LOC
        d_bs_ris = np.linalg.norm(vec_bs_ris, axis=1)
        el_angles = np.arcsin((ris_pos[:, 2] - BS_LOC[2]) / d_bs_ris)
        
        for i in range(MN):
            pl = (LAMBDA / (4 * np.pi * d_bs_ris[i])) ** ALPHA_RIS
            path_phase = np.exp(-1j * 2 * np.pi * d_bs_ris[i] / LAMBDA)
            sv = self._get_bs_steering_vector(el_angles[i])
            H_rd[:, i] = (np.sqrt(pl * G_RIS_ELE_LIN) * path_phase * sv).flatten()
            
        self._H_rd_static = H_rd

    # --- 核心动态更新 ---
    def step_movement(self):
        """更新位置、衰落、生成任务并优化"""
        # 1. 更新位置和 V2I 衰落
        velocities = np.array([v.velocity for v in self.vehicles])
        moved = False
        
        for i, veh in enumerate(self.vehicles):
            dist, reset, lane_changed = veh.update_position(self.dt)
            if reset:
                self.v2v_shadowing_db[i, :] = np.random.normal(0, SHADOW_STD, self.n_total)
                self.v2v_shadowing_db[:, i] = self.v2v_shadowing_db[i, :].T
            if lane_changed:
                # 车道变换时，清除V2V信道缓存（因为位置X坐标改变了）
                self._clear_dynamic_cache()
            if dist > 0:
                veh.update_v2i_fading(self.dt, dist)
                moved = True
        
        # 2. 更新 V2V 衰落
        if moved:
            self._update_v2v_fading(velocities)
            self._clear_dynamic_cache()
            
        # 3. 更新时间
        self.current_time += self.dt

    def _update_v2v_fading(self, velocities):
        # 简化版自回归更新
        v_diff = np.abs(velocities[:, None] - velocities[None, :])
        f_d_matrix = v_diff / LAMBDA
        rho_small = j0(2 * np.pi * f_d_matrix * self.dt)
        
        noise_small = (np.random.normal(0, 1, (self.n_total, self.n_total)) + 
                       1j * np.random.normal(0, 1, (self.n_total, self.n_total))) / np.sqrt(2)
        noise_small = (noise_small + noise_small.T) / 2
        
        self.v2v_h_small = rho_small * self.v2v_h_small + np.sqrt(1 - rho_small**2) * noise_small
        np.fill_diagonal(self.v2v_h_small, 0)
        
        dist_moved = velocities * self.dt
        link_dist_change = (dist_moved[:, None] + dist_moved[None, :]) / 2.0
        rho_shadow = np.exp(-link_dist_change / DECORR_DIST)
        noise_shadow = np.random.normal(0, SHADOW_STD, (self.n_total, self.n_total))
        noise_shadow = (noise_shadow + noise_shadow.T) / 2
        self.v2v_shadowing_db = rho_shadow * self.v2v_shadowing_db + np.sqrt(1 - rho_shadow**2) * noise_shadow

    def _clear_dynamic_cache(self):
        self._cached_h_sr_list = None
        self._cached_h_d_list = None

    def _update_v2i_dynamic_channels(self):
        if self._cached_h_d_list is not None: return

        MN = self.ris.num_elements
        ris_pos = self.ris.element_positions
        h_sr_list = []
        h_d_list = []
        
        for veh in self.vehicles:
            # h_sr
            vec_sr = ris_pos - veh.position
            d_sr = np.linalg.norm(vec_sr, axis=1)
            pl_sr = (LAMBDA / (4 * np.pi * d_sr)) ** ALPHA_RIS
            phase_sr = np.exp(-1j * 2 * np.pi * d_sr / LAMBDA)
            h_sr = np.sqrt(pl_sr) * phase_sr
            h_sr_list.append(h_sr.reshape(MN, 1))
            
            # h_d
            d_d = np.linalg.norm(veh.position - BS_LOC)
            pl_d = (LAMBDA / (4 * np.pi * d_d)) ** ALPHA_DIRECT
            shadow = 10 ** (veh.shadowing_db / 10.0)
            vec_bs_veh = veh.position - BS_LOC
            el_veh = np.arcsin(vec_bs_veh[2] / d_d)
            sv_veh = self._get_bs_steering_vector(el_veh)
            h_d = np.sqrt(pl_d * shadow) * veh.h_small * np.exp(-1j * 2 * np.pi * d_d / LAMBDA) * sv_veh
            h_d_list.append(h_d)
            
        self._cached_h_sr_list = h_sr_list
        self._cached_h_d_list = h_d_list

    # --- 任务处理逻辑 ---

    def find_nearest_service_vehicle(self, task_veh_idx, candidate_set=None):
        """
        寻找最近的可用车辆（可以是任何没有任务的车辆）
        candidate_set: 可选的候选接收车辆集合（索引），如果为 None 则默认所有除自身
        """
        task_veh = self.vehicles[task_veh_idx]
        min_dist = float('inf')
        best_service_idx = -1

        if candidate_set is None:
            candidate_iter = [i for i in range(self.n_total) if i != task_veh_idx]
        else:
            candidate_iter = [i for i in candidate_set if i != task_veh_idx]

        for i in candidate_iter:
            srv_veh = self.vehicles[i]
            dist = np.linalg.norm(task_veh.position - srv_veh.position)
            if dist < min_dist:
                min_dist = dist
                best_service_idx = i

        return best_service_idx, min_dist

    def get_v2i_sinr(self, veh_idx):
        """
        计算特定车辆的 V2I SINR (含 RIS 增强)
        
        RIS辅助V2I通信：基站接收信号为直连信号和RIS级联信号的合成
        - h_d: 车辆到基站的直连信道 (NUM_BS_ANTENNAS x 1)
        - h_sr: 车辆到RIS的信道 (MN x 1)
        - H_rd: RIS到基站的静态信道矩阵 (NUM_BS_ANTENNAS x MN)
        - phi: RIS反射系数向量 (MN x 1)
        - 级联信道: H_casc = H_rd @ diag(phi) @ h_sr = H_rd @ (phi * h_sr)
        - 最终接收信号: h_total = h_d + H_casc
        """
        self._update_v2i_dynamic_channels()
        
        # 获取直连信道和RIS反射信道
        h_d = self._cached_h_d_list[veh_idx]  # (NUM_BS_ANTENNAS x 1)
        h_sr = self._cached_h_sr_list[veh_idx]  # (MN x 1)
        phi = self.ris.get_reflection_matrix()  # (MN,) 一维数组
        
        # 级联信道计算: H_rd @ diag(phi) @ h_sr
        # phi[:, None] 将phi转换为列向量 (MN x 1)
        # phi[:, None] * h_sr 是逐元素相乘，等价于 diag(phi) @ h_sr
        # H_rd_static @ (phi * h_sr) 得到级联信道 (NUM_BS_ANTENNAS x 1)
        H_casc = self._H_rd_static @ (phi[:, None] * h_sr)
        
        # 基站接收信号 = 直连信号 + RIS级联信号
        h_total = h_d + H_casc
        
        # 计算接收功率和SINR
        gain = np.linalg.norm(h_total) ** 2
        # P_rx = P_tx * G_tx * G_rx * |h|^2
        p_rx = TX_POWER_VEHICLE * G_VEH_TX_LIN * G_BS_LIN * gain
        
        # 频分复用，无干扰，只有热噪声
        sinr = p_rx / NOISE_POWER_BS
        return sinr

    def get_v2i_components(self, veh_idx):
        """
        返回直连、级联与合成信号的接收功率/增益，用于测试 RIS 作用
        """
        self._update_v2i_dynamic_channels()

        h_d = self._cached_h_d_list[veh_idx]             # 直连
        h_sr = self._cached_h_sr_list[veh_idx]           # 车 -> RIS
        phi = self.ris.get_reflection_matrix()           # RIS 相位

        H_casc = self._H_rd_static @ (phi[:, None] * h_sr)  # 级联
        h_total = h_d + H_casc                            # 合成

        gain_direct = np.linalg.norm(h_d) ** 2
        gain_casc = np.linalg.norm(H_casc) ** 2
        gain_total = np.linalg.norm(h_total) ** 2

        # 额外链路分量统计
        sr_gain_mean = np.mean(np.abs(h_sr.flatten())**2)      # 车->RIS 平均增益
        rd_gain_mean = np.mean(np.abs(self._H_rd_static)**2)   # RIS->BS 平均增益

        # 接收功率（含收/发增益）
        p_rx_direct = TX_POWER_VEHICLE * G_VEH_TX_LIN * G_BS_LIN * gain_direct
        p_rx_casc = TX_POWER_VEHICLE * G_VEH_TX_LIN * G_BS_LIN * gain_casc
        p_rx_total = TX_POWER_VEHICLE * G_VEH_TX_LIN * G_BS_LIN * gain_total

        sinr_direct = p_rx_direct / NOISE_POWER_BS
        sinr_cascaded = p_rx_casc / NOISE_POWER_BS
        sinr_total = p_rx_total / NOISE_POWER_BS

        return {
            "gain_direct": gain_direct,
            "gain_cascaded": gain_casc,
            "gain_total": gain_total,
            "sr_gain_mean": sr_gain_mean,
            "rd_gain_mean": rd_gain_mean,
            "p_rx_direct": p_rx_direct,
            "p_rx_cascaded": p_rx_casc,
            "p_rx_total": p_rx_total,
            "sinr_direct": sinr_direct,
            "sinr_cascaded": sinr_cascaded,
            "sinr_total": sinr_total
        }

    def get_v2v_sinr(self, tx_idx, rx_idx):
        """
        计算两车之间的 V2V SINR (基于LoS路径损耗模型)
        
        V2V通信使用LoS路径损耗模型：
        - 路径损耗指数: ALPHA_V2V = 2.2 (LoS)
        - 包含阴影衰落和小尺度衰落
        """
        pos_tx = self.vehicles[tx_idx].position
        pos_rx = self.vehicles[rx_idx].position
        dist = np.linalg.norm(pos_tx - pos_rx) + 0.01
        
        # LoS路径损耗模型
        pl = (LAMBDA / (4 * np.pi * dist)) ** ALPHA_V2V
        shadow = 10 ** (self.v2v_shadowing_db[rx_idx, tx_idx] / 10.0)
        fading = np.abs(self.v2v_h_small[rx_idx, tx_idx]) ** 2
        
        gain = pl * shadow * fading
        p_rx = TX_POWER_VEHICLE * G_VEH_TX_LIN * G_VEH_RX_LIN * gain
        
        # V2V接收机噪声功率（基于噪声功率谱密度和带宽）
        return p_rx / NOISE_POWER_V2V

    def calculate_delay_coefficients(self, task_size_bits, snr_v2v, snr_v2i, k_val, return_comm_delays=False):
        """
        计算优化问题中的系数 A_local, A_edge, A_bs
        A_x 代表当 lambda_x = 1 时的该分支总时延
        
        :param return_comm_delays: 如果为True，额外返回通信时延（仅传输部分）
        :return: 如果return_comm_delays=False，返回 (A_local, A_edge, A_bs)
                如果return_comm_delays=True，返回 (A_local, A_edge, A_bs, t_comm_v2v, t_comm_v2i)
        """
        # 1. 本地计算时延 (Full Task)
        # T_loc = Bits * Cycles/Bit / Freq
        t_comp_local = task_size_bits * CYCLES_PER_BIT / CPU_FREQ_LOCAL
        A_local = t_comp_local
        
        # 2. 边缘 (V2V) 时延 = 传输 + 计算
        # 语义相似度 delta
        delta_v2v = self.sem_model.get_similarity(snr_v2v, k_val)
        delta_v2v = max(0.001, delta_v2v)
        
        # 语义传输时延 (参考公式)
        # num_sentences = task_size_bits / AH
        # T_trans = (num_sentences * k * Aw) / (B * delta)
        # 简化计算：直接计算 effective rate
        # Rate_sem = (B * As) / (Aw * K * delta) ? 这里使用简化的 bits 传输概念
        # 我们使用 bits 传输时间 / 语义增益
        # 或直接套用 lambda_lp_solver 期望的"满载时延"
        
        # 假设语义传输的数据量压缩比
        # 原始公式含义：传输语义符号所需时间
        num_sentences = task_size_bits / AH
        sym_to_send = num_sentences * k_val * AW
        # 假设每个 symbol 占用 1/Bandwidth 秒 (粗略估计) 或者 1 bit/Hz/s
        # T_trans = sym_to_send / (BANDWIDTH * delta)  (delta 影响有效吞吐)
        t_trans_v2v = sym_to_send / (BANDWIDTH * delta_v2v)
        
        t_comp_edge = task_size_bits * CYCLES_PER_BIT / CPU_FREQ_SERVICE
        A_edge = t_trans_v2v + t_comp_edge
        
        # 3. 基站 (V2I) 时延 = 传输 + 计算
        delta_v2i = self.sem_model.get_similarity(snr_v2i, k_val)
        delta_v2i = max(0.001, delta_v2i)
        
        t_trans_v2i = sym_to_send / (BANDWIDTH * delta_v2i)
        t_comp_bs = task_size_bits * CYCLES_PER_BIT / CPU_FREQ_BS
        A_bs = t_trans_v2i + t_comp_bs
        
        if return_comm_delays:
            return A_local, A_edge, A_bs, t_trans_v2v, t_trans_v2i
        return A_local, A_edge, A_bs

    def process_offloading(self):
        """
        执行：任务生成 -> 服务匹配 -> SINR计算 -> 卸载优化
        返回本时隙所有生成任务的处理结果
        
        注意：任务服从泊松到达过程，可能在同一时隙生成多个任务
        """
        results = []
        
        # 随机设置 RIS 相位 (因为本例不包含RIS优化)
        random_phases = np.random.uniform(0, 2*np.pi, self.ris.num_elements)
        self.ris.set_phases(random_phases)
        
        # 可用接收车辆集合：初始为所有车辆，后续移除有任务的车辆和已被占用的接收车辆
        available_receivers = set(range(self.n_total))
        
        # 遍历任务车辆
        for i in range(self.n_task_veh):
            veh = self.vehicles[i]
            
            # 1. 任务生成 (Poisson) - 正确处理多个任务
            num_tasks = np.random.poisson(POISSON_RATE * self.dt)
            
            # 处理本时隙生成的所有任务
            for _ in range(num_tasks):
                task = Task(self.task_counter, veh.id, TASK_SIZE_MBIT, self.current_time)
                self.task_counter += 1
                
                # 该车辆有任务，不能作为接收端
                if i in available_receivers:
                    available_receivers.discard(i)
                
                # 2. 服务节点匹配 (寻找无任务且未被占用的车辆)
                srv_idx, dist = self.find_nearest_service_vehicle(i, candidate_set=available_receivers)
                if srv_idx != -1:
                    # 该接收车辆被占用，移出可用集合，保证“一车一任务”
                    available_receivers.discard(srv_idx)
                
                # 3. 计算物理层状态 (SINR)
                # V2V
                if srv_idx != -1:
                    snr_v2v = self.get_v2v_sinr(i, srv_idx)
                else:
                    snr_v2v = 1e-10 # 无连接
                
                # V2I (RIS辅助，基站接收信号为直连和级联合成)
                snr_v2i = self.get_v2i_sinr(i)
                
                # 4. 准备优化参数
                k_val = SEMANTIC_K_DEFAULT
                A_local, A_edge, A_bs = self.calculate_delay_coefficients(
                    task.size_bits, snr_v2v, snr_v2i, k_val
                )
                
                # 5. 求解最优 Lambda
                if HAS_SOLVER:
                    lam_l, lam_e, lam_b, t_opt = compute_optimal_lambda(
                        A_local, A_edge, A_bs, prefer="closed_form"
                    )
                else:
                    # 备选：平均分配
                    lam_l, lam_e, lam_b = 0.33, 0.33, 0.34
                    t_opt = max(A_local*lam_l, A_edge*lam_e, A_bs*lam_b)
                
                # 记录结果
                task.lambda_local = lam_l
                task.lambda_edge = lam_e
                task.lambda_bs = lam_b
                task.total_delay = t_opt
                
                results.append({
                    "task_id": task.id,
                    "veh_id": veh.id,
                    "srv_id": self.vehicles[srv_idx].id if srv_idx != -1 else -1,
                    "snr_v2v_db": 10*np.log10(snr_v2v),
                    "snr_v2i_db": 10*np.log10(snr_v2i),
                    "lambda": (lam_l, lam_e, lam_b),
                    "delay": t_opt
                })
                
        return results

# ================= 运行测试 =================

if __name__ == "__main__":
    env = Environ(n_vehicles=8, dt=0.1)

    print("=== RIS Gain Test (Single Snapshot) ===")
    env.step_movement()  # 更新位置与信道

    test_idx = 0
    comps = env.get_v2i_components(test_idx)
    print(f"Veh {test_idx} -> BS (with RIS)")
    print(f"  Gain_direct   = {10*np.log10(comps['gain_direct']+1e-15):.2f} dB")
    print(f"  Gain_cascaded = {10*np.log10(comps['gain_cascaded']+1e-15):.2f} dB")
    print(f"  Gain_total    = {10*np.log10(comps['gain_total']+1e-15):.2f} dB")
    print(f"  SR_gain_mean  = {10*np.log10(comps['sr_gain_mean']+1e-15):.2f} dB  (Veh->RIS mean element gain)")
    print(f"  RD_gain_mean  = {10*np.log10(comps['rd_gain_mean']+1e-15):.2f} dB  (RIS->BS mean element gain)")
    print(f"  P_rx_direct   = {10*np.log10(comps['p_rx_direct']*1e3+1e-15):.2f} dBm")
    print(f"  P_rx_cascaded = {10*np.log10(comps['p_rx_cascaded']*1e3+1e-15):.2f} dBm")
    print(f"  P_rx_total    = {10*np.log10(comps['p_rx_total']*1e3+1e-15):.2f} dBm")
    print(f"  SNR_direct    = {10*np.log10(comps['sinr_direct']+1e-15):.2f} dB")
    print(f"  SNR_cascaded  = {10*np.log10(comps['sinr_cascaded']+1e-15):.2f} dB")
    print(f"  SINR_total    = {10*np.log10(comps['sinr_total']+1e-15):.2f} dB")

    print("\n=== Simulation Start (Physics + Task Offloading) ===")
    
    total_steps = 20
    for step in range(total_steps):
        # 1. 物理移动与信道更新
        env.step_movement()
        
        # 2. 任务生成与卸载决策
        step_results = env.process_offloading()
        
        if step_results:
            print(f"\n[Time {env.current_time:.1f}s] Generated {len(step_results)} tasks:")
            for res in step_results:
                lam = res['lambda']
                print(f"  Task {res['task_id']} (Veh {res['veh_id']} -> Srv {res['srv_id']}):")
                print(f"    SNRs: V2V={res['snr_v2v_db']:.1f}dB, V2I={res['snr_v2i_db']:.1f}dB")
                print(f"    Offloading: Loc={lam[0]:.2f}, Edge={lam[1]:.2f}, BS={lam[2]:.2f}")
                print(f"    MinMax Delay: {res['delay']*1000:.2f} ms")