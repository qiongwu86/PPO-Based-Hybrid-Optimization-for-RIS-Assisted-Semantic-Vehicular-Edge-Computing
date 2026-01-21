import numpy as np
import copy
import math

# 导入环境配置
from environment import Environ, TASK_SIZE_MBIT

# ================= GA 配置参数 (可快捷调整) =================
# 遗传算法超参数
GA_POP_SIZE = 50          # 种群大小
GA_GENERATIONS = 50       # 迭代次数
GA_CROSSOVER_RATE = 0.8   # 交叉概率
GA_MUTATION_RATE = 0.1    # 变异概率
GA_ELITE_SIZE = 2         # 精英保留数量

# 变异参数
MUTATION_RIS_NOISE = 0.5   # RIS相位变异噪声标准差
MUTATION_K_RANGE = (-2, 3) # K值变异范围
MUTATION_LAM_NOISE = 0.2   # Lambda权重变异噪声标准差
MUTATION_GENE_RATIO = 0.1  # 每次变异时变异的基因比例

# 语义相似度约束参数
SEM_SIM_THRESHOLD = 0.9    # 语义相似度阈值（低于此值将受到惩罚）
SEM_PENALTY_SCALE = 5.0    # 惩罚系数（越大约束越强，惩罚 = SCALE * (THRESHOLD - actual_similarity)）
USE_SEMANTIC_PENALTY = True  # 是否启用语义相似度惩罚

# ================= 遗传算法求解器 =================

class GeneticAlgorithmSolver:
    def __init__(self, 
                 n_vehicles=None,
                 ris_row=None,
                 ris_col=None,
                 pop_size=None, 
                 generations=None, 
                 crossover_rate=None, 
                 mutation_rate=None, 
                 elite_size=None):
        """
        初始化遗传算法求解器
        
        :param n_vehicles: 车辆数量（如果为None，将从环境推断）
        :param ris_row: RIS行数（如果为None，将从环境推断）
        :param ris_col: RIS列数（如果为None，将从环境推断）
        :param pop_size: 种群大小
        :param generations: 迭代次数
        :param crossover_rate: 交叉概率
        :param mutation_rate: 变异概率
        :param elite_size: 精英保留数量
        """
        # 使用配置参数或默认值
        self.pop_size = pop_size if pop_size is not None else GA_POP_SIZE
        self.generations = generations if generations is not None else GA_GENERATIONS
        self.crossover_rate = crossover_rate if crossover_rate is not None else GA_CROSSOVER_RATE
        self.mutation_rate = mutation_rate if mutation_rate is not None else GA_MUTATION_RATE
        self.elite_size = elite_size if elite_size is not None else GA_ELITE_SIZE
        
        # 存储配置（如果提供）
        self.n_vehicles = n_vehicles
        self.ris_row = ris_row
        self.ris_col = ris_col
        self.n_ris_elements = None  # 将在首次使用时从环境推断
        
        # 变异参数
        self.mutation_ris_noise = MUTATION_RIS_NOISE
        self.mutation_k_range = MUTATION_K_RANGE
        self.mutation_lam_noise = MUTATION_LAM_NOISE
        self.mutation_gene_ratio = MUTATION_GENE_RATIO
        
        # 语义相似度约束参数
        self.sem_sim_threshold = SEM_SIM_THRESHOLD
        self.sem_penalty_scale = SEM_PENALTY_SCALE
        self.use_semantic_penalty = USE_SEMANTIC_PENALTY
        
        # 染色体长度将在首次使用时计算
        self.idx_ris_end = None
        self.idx_k_end = None
        self.idx_lam_end = None
        self.chromosome_len = None

    def _infer_config_from_env(self, env):
        """从环境对象推断配置"""
        if self.n_vehicles is None:
            self.n_vehicles = env.n_total
        if self.ris_row is None:
            self.ris_row = env.ris.M
        if self.ris_col is None:
            self.ris_col = env.ris.N
        
        self.n_ris_elements = self.ris_row * self.ris_col
        
        # 计算染色体索引
        self.idx_ris_end = self.n_ris_elements
        self.idx_k_end = self.n_ris_elements + self.n_vehicles
        self.idx_lam_end = self.n_ris_elements + self.n_vehicles + (3 * self.n_vehicles)
        self.chromosome_len = self.idx_lam_end

    def init_population(self):
        """初始化种群"""
        if self.chromosome_len is None:
            raise ValueError("Configuration not initialized. Call _infer_config_from_env first or provide n_vehicles, ris_row, ris_col.")
        
        pop = np.zeros((self.pop_size, self.chromosome_len))
        
        # 1. RIS 相位 [0, 2pi]
        pop[:, :self.idx_ris_end] = np.random.uniform(0, 2*np.pi, (self.pop_size, self.n_ris_elements))
        
        # 2. K值 [1, 20]
        pop[:, self.idx_ris_end:self.idx_k_end] = np.random.randint(1, 21, (self.pop_size, self.n_vehicles))
        
        # 3. Lambda 权重 [0, 1]
        pop[:, self.idx_k_end:] = np.random.uniform(0, 1, (self.pop_size, 3 * self.n_vehicles))
        
        return pop

    def decode_individual(self, individual):
        """
        将染色体解码为具体的物理意义参数
        """
        # A. RIS 相位
        ris_phases = individual[:self.idx_ris_end]
        
        # B. K 值 (取整)
        k_values = np.round(individual[self.idx_ris_end:self.idx_k_end]).astype(int)
        k_values = np.clip(k_values, 1, 20)
        
        # C. Lambda 比例 (归一化)
        raw_lam = individual[self.idx_k_end:].reshape(self.n_vehicles, 3)
        # 防止除零
        sum_lam = np.sum(raw_lam, axis=1, keepdims=True) + 1e-9
        lambdas = raw_lam / sum_lam # (N_Veh, 3) -> [Local, Edge, BS]
        
        return ris_phases, k_values, lambdas

    def evaluate_fitness(self, population, env_snapshot, return_details=False):
        """
        计算适应度
        注意：我们需要在这个函数里"临时"修改环境的RIS相位来计算SINR，
        但不能调用 step_movement，因为GA是在同一个时间步内搜索。
        
        :param return_details: 如果为True，返回(适应度, 详细信息)的元组
        :return: 如果return_details=False，返回适应度数组；否则返回(适应度数组, 详细信息列表)
        """
        # 首次使用时推断配置
        if self.chromosome_len is None:
            self._infer_config_from_env(env_snapshot)
        
        fitness_scores = []
        delay_details = [] if return_details else None
        
        # 获取实际种群大小（可能小于self.pop_size）
        actual_pop_size = population.shape[0]
        
        # 备份原始RIS相位，以免影响外部环境
        original_phases = env_snapshot.ris.phases.copy()
        
        # 任务大小 (bits)
        task_bits = TASK_SIZE_MBIT * 1e6
        
        for i in range(actual_pop_size):
            ind = population[i]
            ris_phases, k_values, lambdas = self.decode_individual(ind)
            
            # 1. 设置环境 RIS (为了计算 V2I SINR)
            env_snapshot.ris.set_phases(ris_phases)
            # 注意：环境类中获取 SINR 的函数会用到当前 RIS 状态
            
            max_delays = []
            semantic_penalty = 0.0
            
            # 2. 遍历所有车辆计算时延和语义相似度
            for v_idx in range(self.n_vehicles):
                k_val = k_values[v_idx]
                lam_loc, lam_edge, lam_bs = lambdas[v_idx]
                
                # 获取 SINR
                srv_idx, _ = env_snapshot.find_nearest_service_vehicle(v_idx)
                
                snr_v2v = 1e-10
                if srv_idx != -1:
                    snr_v2v = env_snapshot.get_v2v_sinr(v_idx, srv_idx)
                
                # 获取 V2I SINR (受 RIS 影响)
                snr_v2i = env_snapshot.get_v2i_sinr(v_idx)
                
                # 计算 A 参数 (单位负载时延)
                A_l, A_e, A_b = env_snapshot.calculate_delay_coefficients(
                    task_bits, snr_v2v, snr_v2i, k_val
                )
                
                # 计算实际时延 (三者并行，取最大值)
                d_loc = A_l * lam_loc
                d_edge = A_e * lam_edge
                d_bs = A_b * lam_bs
                
                # 车辆的总时延取决于最慢的那个分流
                veh_total_delay = max(d_loc, d_edge, d_bs)
                max_delays.append(veh_total_delay)
                
                # 语义相似度惩罚（针对 V2I 链路）
                if self.use_semantic_penalty:
                    delta_v2i = env_snapshot.sem_model.get_similarity(snr_v2i, k_val)
                    
                    # 如果语义相似度低于阈值，添加惩罚
                    if delta_v2i < self.sem_sim_threshold:
                        penalty = self.sem_penalty_scale * (self.sem_sim_threshold - delta_v2i)
                        semantic_penalty += penalty
            
            # 3. 系统目标：最小化所有车辆中的最大时延 (Min-Max)
            system_objective = np.max(max_delays)
            
            # 4. 总目标 = 时延 + 语义惩罚
            total_objective = system_objective + semantic_penalty
            
            # 适应度 = -目标值 (GA 是最大化适应度)
            fitness_scores.append(-total_objective)
            
            if return_details:
                delay_details.append({
                    'real_delay': system_objective,
                    'penalty': semantic_penalty,
                    'total_objective': total_objective
                })
        
        # 恢复环境
        env_snapshot.ris.set_phases(original_phases)
        
        if return_details:
            return np.array(fitness_scores), delay_details
        return np.array(fitness_scores)

    def select(self, population, fitness):
        """锦标赛选择"""
        new_pop = np.zeros_like(population)
        # 精英保留
        sorted_indices = np.argsort(fitness)[::-1] # 降序
        new_pop[:self.elite_size] = population[sorted_indices[:self.elite_size]]
        
        for i in range(self.elite_size, self.pop_size):
            # 随机选 3 个打比赛
            candidates_idx = np.random.randint(0, self.pop_size, 3)
            best_idx = candidates_idx[np.argmax(fitness[candidates_idx])]
            new_pop[i] = population[best_idx]
            
        return new_pop

    def crossover(self, population):
        """均匀交叉"""
        new_pop = population.copy()
        # 从精英之后开始交叉
        for i in range(self.elite_size, self.pop_size, 2):
            if i + 1 >= self.pop_size: 
                break
            
            if np.random.rand() < self.crossover_rate:
                # 随机生成掩码
                mask = np.random.rand(self.chromosome_len) < 0.5
                parent1 = population[i]
                parent2 = population[i+1]
                
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
                
                new_pop[i] = child1
                new_pop[i+1] = child2
        return new_pop

    def mutate(self, population):
        """高斯变异"""
        # 同样保留精英不因变异丢失
        for i in range(self.elite_size, self.pop_size):
            if np.random.rand() < self.mutation_rate:
                # 对个体的某些基因添加噪声
                mutation_mask = np.random.rand(self.chromosome_len) < self.mutation_gene_ratio
                
                # RIS 相位变异
                noise_ris = np.random.normal(0, self.mutation_ris_noise, self.n_ris_elements)
                
                # K 值变异 (整数跳变)
                noise_k = np.random.randint(self.mutation_k_range[0], self.mutation_k_range[1] + 1, self.n_vehicles)
                
                # Lambda 权重变异
                noise_lam = np.random.normal(0, self.mutation_lam_noise, 3 * self.n_vehicles)
                
                noise_total = np.concatenate([noise_ris, noise_k, noise_lam])
                
                population[i] += mutation_mask * noise_total
                
                # 边界截断修复
                # RIS
                population[i, :self.idx_ris_end] = np.mod(population[i, :self.idx_ris_end], 2*np.pi)
                # K
                population[i, self.idx_ris_end:self.idx_k_end] = np.clip(
                    population[i, self.idx_ris_end:self.idx_k_end], 1, 20
                )
                # Lambda Weights
                population[i, self.idx_k_end:] = np.clip(
                    population[i, self.idx_k_end:], 0.01, 1.0
                )
                
        return population

    def solve(self, env_snapshot, verbose=False):
        """
        主求解函数
        :param env_snapshot: 当前环境对象 (GA不会修改其内部物理状态)
        :param verbose: 是否打印详细信息
        :return: best_ris_phases, best_k_values, best_lambdas, best_delay
        """
        # 首次使用时推断配置
        if self.chromosome_len is None:
            self._infer_config_from_env(env_snapshot)
        
        population = self.init_population()
        
        best_fitness_history = []
        best_solution = None
        best_score = -float('inf')
        best_real_delay = None
        best_penalty = None
        
        for gen in range(self.generations):
            # 在verbose模式下，获取详细信息以显示真实延迟
            if verbose and gen % 10 == 0:
                fitness, details = self.evaluate_fitness(population, env_snapshot, return_details=True)
            else:
                fitness = self.evaluate_fitness(population, env_snapshot, return_details=False)
                details = None
            
            # 记录本代最佳
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_score:
                best_score = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
                if details is not None:
                    best_real_delay = details[gen_best_idx]['real_delay']
                    best_penalty = details[gen_best_idx]['penalty']
            
            best_fitness_history.append(best_score)
            
            if verbose and gen % 10 == 0:
                if details is not None:
                    gen_best_real = details[gen_best_idx]['real_delay']
                    gen_best_penalty = details[gen_best_idx]['penalty']
                    if self.use_semantic_penalty and gen_best_penalty > 0:
                        print(f"GA Gen {gen}: Real Delay = {gen_best_real:.4f} s, Penalty = {gen_best_penalty:.4f}, Total = {-best_score:.4f} s")
                    else:
                        print(f"GA Gen {gen}: Real Delay = {gen_best_real:.4f} s (No penalty)")
                else:
                    print(f"GA Gen {gen}: Total Objective = {-best_score:.4f} s")
            
            # 进化操作
            population = self.select(population, fitness)
            population = self.crossover(population)
            population = self.mutate(population)
        
        # 解码最终最优解
        best_ris, best_k, best_lam = self.decode_individual(best_solution)
        
        # 如果需要返回真实延迟，重新计算一次（确保准确性）
        if verbose or best_real_delay is None:
            _, final_details = self.evaluate_fitness(np.array([best_solution]), env_snapshot, return_details=True)
            if final_details:
                best_real_delay = final_details[0]['real_delay']
                best_penalty = final_details[0]['penalty']
        
        # 返回：RIS相位, K值, Lambda权重, 真实延迟, 惩罚值, 总目标值
        return best_ris, best_k, best_lam, best_real_delay, best_penalty, -best_score

# ================= 单元测试与使用示例 =================
if __name__ == "__main__":
    # ================= 测试配置 (可快捷调整) =================
    TEST_N_VEHICLES = 15    # 测试车辆数
    TEST_RIS_ROW = 6        # 测试RIS行数
    TEST_RIS_COL = 6       # 测试RIS列数
    
    # 1. 创建环境
    env = Environ(n_vehicles=TEST_N_VEHICLES, dt=0.1)
    
    # 修改RIS尺寸（如果需要）
    if TEST_RIS_ROW != env.ris.M or TEST_RIS_COL != env.ris.N:
        env.ris.M = TEST_RIS_ROW
        env.ris.N = TEST_RIS_COL
        env.ris.num_elements = TEST_RIS_ROW * TEST_RIS_COL
        env.ris.element_positions = np.zeros((env.ris.num_elements, 3))
        env.ris.phases = np.zeros(env.ris.num_elements)
        env.ris._init_geometry_yz()
        env._compute_static_channel()
    
    # 2. 初始化 GA 求解器
    # 方式1: 自动从环境推断配置
    ga_solver = GeneticAlgorithmSolver(
        pop_size=50, 
        generations=50
    )
    
    # 方式2: 手动指定配置（可选）
    # ga_solver = GeneticAlgorithmSolver(
    #     n_vehicles=TEST_N_VEHICLES,
    #     ris_row=TEST_RIS_ROW,
    #     ris_col=TEST_RIS_COL,
    #     pop_size=30,
    #     generations=20
    # )
    
    print("="*60)
    print("Running GA Optimization for one step...")
    print(f"Configuration: {TEST_N_VEHICLES} vehicles, RIS {TEST_RIS_ROW}×{TEST_RIS_COL}")
    print("="*60)
    
    # 3. 运行 GA
    ris, k, lam, real_delay, penalty, total_obj = ga_solver.solve(env, verbose=True)
    
    print("\n" + "="*60)
    print("=== Optimization Result ===")
    print(f"Real Delay (Min-Max): {real_delay:.4f} s")
    if penalty > 0:
        print(f"Semantic Penalty:     {penalty:.4f}")
        print(f"Total Objective:      {total_obj:.4f} s (Real Delay + Penalty)")
    else:
        print(f"Semantic Penalty:     0.0000 (All vehicles meet threshold)")
        print(f"Total Objective:      {total_obj:.4f} s")
    print(f"Sample K values: {k[:5]}")
    print(f"Sample Lambda (Veh 0): Local={lam[0][0]:.3f}, Edge={lam[0][1]:.3f}, BS={lam[0][2]:.3f}")
    print("="*60)
    
    # 4. 将结果应用到环境 (如果是实际仿真循环)
    # env.ris.set_phases(ris)
    # env.step_movement()
