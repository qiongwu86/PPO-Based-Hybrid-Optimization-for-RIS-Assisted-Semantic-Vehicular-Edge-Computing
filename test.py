import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
import re
from collections import defaultdict

# 导入必要的类和函数
from MAPPO import SpecificStateEnv, PPO, calculate_dims
from GA import GeneticAlgorithmSolver
from QPSO import QPSOSolver
import environment as env_cfg
from environment import Environ, BANDWIDTH, AH, AW

# ================= 测试配置 =================
PPO_TEST_STEPS = 200  # PPO测试总时间步
GA_TEST_STEPS = 50   # GA测试总时间步
QPSO_TEST_STEPS = 50 # QPSO测试总时间步
OPTIMIZATION_STEPS = 50  # GA/QPSO在每个时间步的优化迭代次数
TASK_SIZE_MBIT = 0.4  # 假设任务大小 (需与环境一致)
RANDOM_SEED = 42  # 固定随机种子确保可重复性

# 任务到达率与发射功率扫描配置（可按需修改）
ARRIVAL_RATES = [4.0, 6.0, 8.0, 10.0, 12.0]  # 任务到达率 (tasks/sec)
TX_POWER_LIST = [0.1, 0.15, 0.2, 0.25, 0.3]        # 车辆发射功率 (Watts)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_model_filename(filename):
    """
    解析模型文件名，提取配置信息
    返回: (n_vehicles, ris_row, ris_col) 或 None
    """
    # 匹配模式1: ppo_specific_veh15_RIS8x8.pth
    match1 = re.match(r'ppo_specific_veh(\d+)_RIS(\d+)x(\d+)\.pth', filename)
    if match1:
        return int(match1.group(1)), int(match1.group(2)), int(match1.group(3))
    
    # 匹配模式2: ppo_specific_veh15.pth (默认RIS 6x6)
    match2 = re.match(r'ppo_specific_veh(\d+)\.pth', filename)
    if match2:
        return int(match2.group(1)), 6, 6
    
    # 匹配模式3: ppo_specific.pth (10车，RIS 6x6)
    if filename == 'ppo_specific.pth':
        return 10, 6, 6
    
    return None

def find_model_files():
    """查找所有模型文件并分类"""
    all_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    # 组1: RIS 6x6，不同车辆数目
    group1 = []  # [(filename, n_vehicles, ris_row, ris_col), ...]
    
    # 组2: 固定15辆车，不同RIS大小
    group2 = []  # [(filename, n_vehicles, ris_row, ris_col), ...]
    
    for filename in all_files:
        config = parse_model_filename(filename)
        if config is None:
            continue
        
        n_veh, ris_r, ris_c = config
        
        # 组1: RIS 6x6，不同车辆数目
        if ris_r == 6 and ris_c == 6:
            group1.append((filename, n_veh, ris_r, ris_c))
        
        # 组2: 固定15辆车，不同RIS大小（包括6x6）
        if n_veh == 15:
            group2.append((filename, n_veh, ris_r, ris_c))
    
    # 排序
    group1.sort(key=lambda x: x[1])  # 按车辆数排序
    group2.sort(key=lambda x: (x[2], x[3]))  # 按RIS大小排序
    
    return group1, group2

def run_ppo_test(model_path, n_vehicles, ris_row, ris_col, test_steps=PPO_TEST_STEPS):
    """
    运行PPO模型的测试
    返回: (平均时延, 详细数据记录列表)
    """
    set_seed(RANDOM_SEED)
    
    # 计算维度
    state_dim, action_dim, n_ris_elements = calculate_dims(n_vehicles, ris_row, ris_col)
    
    # 初始化环境和智能体
    env_wrapper = SpecificStateEnv(n_vehicles=n_vehicles, ris_row=ris_row, ris_col=ris_col)
    ppo_agent = PPO(state_dim, action_dim)
    
    # 加载模型权重
    if not os.path.exists(model_path):
        print(f"PPO Model file not found: {model_path}")
        return None, None
    
    try:
        ppo_agent.policy_old.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading PPO model: {e}")
        return None, None
    
    ppo_agent.policy_old.eval()
    
    records = []  # 详细数据记录
    delays = []
    state = env_wrapper.reset()
    
    for t in range(test_steps):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            action_mean = ppo_agent.policy_old.actor(ppo_agent.policy_old.shared_net(state_tensor))
            action = action_mean.cpu().numpy().flatten()
        
        ris_act = action[:n_ris_elements]
        ris_phases = (ris_act + 1) * np.pi 
        env_wrapper.env.ris.set_phases(ris_phases)
        env_wrapper.current_ris_phases = ris_phases
        
        k_act = action[n_ris_elements:]
        k_values = np.round((k_act + 1) / 2 * 19 + 1).astype(int)
        k_values = np.clip(k_values, 1, 20)
        
        env_wrapper.env.step_movement()
        
        for i in range(n_vehicles):
            k_val = k_values[i]
            srv_idx, _ = env_wrapper.env.find_nearest_service_vehicle(i)
            
            snr_v2v = 1e-10
            if srv_idx != -1:
                snr_v2v = env_wrapper.env.get_v2v_sinr(i, srv_idx)
            snr_v2i = env_wrapper.env.get_v2i_sinr(i)
            
            task_bits = TASK_SIZE_MBIT * 1e6
            # 获取延迟系数和通信时延（基于整个任务）
            A_l, A_e, A_b, t_comm_v2v_full, t_comm_v2i_full = env_wrapper.env.calculate_delay_coefficients(
                task_bits, snr_v2v, snr_v2i, k_val, return_comm_delays=True
            )
            
            # LP 优化求解 lambda
            try:
                from lambda_lp_solver import compute_optimal_lambda
                lam_l, lam_e, lam_b, t_opt = compute_optimal_lambda(A_l, A_e, A_b, prefer="closed_form")
            except Exception:
                lam_l, lam_e, lam_b = 1/3, 1/3, 1/3
                t_opt = max(A_l, A_e, A_b)
            
            # 通信时延（V2V 和 V2I，仅传输部分）
            # 注意：环境返回的是基于整个任务的传输时延，需要按lambda比例缩放
            t_comm_v2v = t_comm_v2v_full * lam_e if lam_e > 0 else 0.0
            t_comm_v2i = t_comm_v2i_full * lam_b if lam_b > 0 else 0.0
            
            delays.append(t_opt)
            
            # 记录详细数据
            records.append({
                'method': 'PPO',
                'step': t,
                'vehicle_id': i,
                'delay_total': t_opt,
                'delay_local': A_l * lam_l,
                'delay_edge': A_e * lam_e,
                'delay_bs': A_b * lam_b,
                'delay_comm_v2v': t_comm_v2v,
                'delay_comm_v2i': t_comm_v2i,
                'k_value': k_val,
                'lambda_local': lam_l,
                'lambda_edge': lam_e,
                'lambda_bs': lam_b,
                'snr_v2v_db': 10 * np.log10(snr_v2v + 1e-15),
                'snr_v2i_db': 10 * np.log10(snr_v2i + 1e-15),
                'A_local': A_l,
                'A_edge': A_e,
                'A_bs': A_b
            })
        
        state = env_wrapper._get_state()
    
    avg_delay = np.mean(delays)
    return avg_delay, records

def run_ga_test(n_vehicles, ris_row, ris_col, test_steps=GA_TEST_STEPS):
    """
    运行GA优化测试
    返回: (平均时延, 详细数据记录列表)
    """
    set_seed(RANDOM_SEED)
    
    env = Environ(n_vehicles=n_vehicles, dt=0.1)
    
    # 修改RIS尺寸
    if ris_row != env.ris.M or ris_col != env.ris.N:
        env.ris.M = ris_row
        env.ris.N = ris_col
        env.ris.num_elements = ris_row * ris_col
        env.ris.element_positions = np.zeros((env.ris.num_elements, 3))
        env.ris.phases = np.zeros(env.ris.num_elements)
        env.ris._init_geometry_yz()
        env._compute_static_channel()
    
    ga_solver = GeneticAlgorithmSolver(
        n_vehicles=n_vehicles,
        ris_row=ris_row,
        ris_col=ris_col,
        pop_size=30,
        generations=OPTIMIZATION_STEPS
    )
    
    records = []  # 详细数据记录
    delays = []
    
    for t in range(test_steps):
        env.step_movement()
        
        # 在每个时间步运行GA优化
        ris, k, lam, real_delay, penalty, total_obj = ga_solver.solve(env, verbose=False)
        
        # 应用优化结果
        env.ris.set_phases(ris)
        
        # 计算所有车辆的时延
        task_bits = TASK_SIZE_MBIT * 1e6
        
        for i in range(n_vehicles):
            k_val = k[i]
            lam_loc, lam_edge, lam_bs = lam[i]
            
            srv_idx, _ = env.find_nearest_service_vehicle(i)
            snr_v2v = 1e-10
            if srv_idx != -1:
                snr_v2v = env.get_v2v_sinr(i, srv_idx)
            snr_v2i = env.get_v2i_sinr(i)
            
            # 获取延迟系数和通信时延（基于整个任务）
            A_l, A_e, A_b, t_comm_v2v_full, t_comm_v2i_full = env.calculate_delay_coefficients(
                task_bits, snr_v2v, snr_v2i, k_val, return_comm_delays=True
            )
            
            d_loc = A_l * lam_loc
            d_edge = A_e * lam_edge
            d_bs = A_b * lam_bs
            t_opt = max(d_loc, d_edge, d_bs)
            
            # 通信时延（V2V 和 V2I，仅传输部分）
            # 注意：环境返回的是基于整个任务的传输时延，需要按lambda比例缩放
            t_comm_v2v = t_comm_v2v_full * lam_edge if lam_edge > 0 else 0.0
            t_comm_v2i = t_comm_v2i_full * lam_bs if lam_bs > 0 else 0.0
            
            delays.append(t_opt)
            
            # 记录详细数据
            records.append({
                'method': 'GA',
                'step': t,
                'vehicle_id': i,
                'delay_total': t_opt,
                'delay_local': d_loc,
                'delay_edge': d_edge,
                'delay_bs': d_bs,
                'delay_comm_v2v': t_comm_v2v,
                'delay_comm_v2i': t_comm_v2i,
                'k_value': k_val,
                'lambda_local': lam_loc,
                'lambda_edge': lam_edge,
                'lambda_bs': lam_bs,
                'snr_v2v_db': 10 * np.log10(snr_v2v + 1e-15),
                'snr_v2i_db': 10 * np.log10(snr_v2i + 1e-15),
                'A_local': A_l,
                'A_edge': A_e,
                'A_bs': A_b
            })
    
    avg_delay = np.mean(delays)
    return avg_delay, records

def run_qpso_test(n_vehicles, ris_row, ris_col, test_steps=QPSO_TEST_STEPS):
    """
    运行QPSO优化测试
    返回: (平均时延, 详细数据记录列表)
    """
    set_seed(RANDOM_SEED)
    
    env = Environ(n_vehicles=n_vehicles, dt=0.1)
    
    # 修改RIS尺寸
    if ris_row != env.ris.M or ris_col != env.ris.N:
        env.ris.M = ris_row
        env.ris.N = ris_col
        env.ris.num_elements = ris_row * ris_col
        env.ris.element_positions = np.zeros((env.ris.num_elements, 3))
        env.ris.phases = np.zeros(env.ris.num_elements)
        env.ris._init_geometry_yz()
        env._compute_static_channel()
    
    qpso_solver = QPSOSolver(
        n_vehicles=n_vehicles,
        ris_row=ris_row,
        ris_col=ris_col,
        pop_size=30,
        iterations=OPTIMIZATION_STEPS
    )
    
    records = []  # 详细数据记录
    delays = []
    
    for t in range(test_steps):
        env.step_movement()
        
        # 在每个时间步运行QPSO优化
        ris, k, lam, real_delay, penalty, total_obj = qpso_solver.solve(env, verbose=False)
        
        # 应用优化结果
        env.ris.set_phases(ris)
        
        # 计算所有车辆的时延
        task_bits = TASK_SIZE_MBIT * 1e6
        
        for i in range(n_vehicles):
            k_val = k[i]
            lam_loc, lam_edge, lam_bs = lam[i]
            
            srv_idx, _ = env.find_nearest_service_vehicle(i)
            snr_v2v = 1e-10
            if srv_idx != -1:
                snr_v2v = env.get_v2v_sinr(i, srv_idx)
            snr_v2i = env.get_v2i_sinr(i)
            
            # 获取延迟系数和通信时延（基于整个任务）
            A_l, A_e, A_b, t_comm_v2v_full, t_comm_v2i_full = env.calculate_delay_coefficients(
                task_bits, snr_v2v, snr_v2i, k_val, return_comm_delays=True
            )
            
            d_loc = A_l * lam_loc
            d_edge = A_e * lam_edge
            d_bs = A_b * lam_bs
            t_opt = max(d_loc, d_edge, d_bs)
            
            # 通信时延（V2V 和 V2I，仅传输部分）
            # 注意：环境返回的是基于整个任务的传输时延，需要按lambda比例缩放
            t_comm_v2v = t_comm_v2v_full * lam_edge if lam_edge > 0 else 0.0
            t_comm_v2i = t_comm_v2i_full * lam_bs if lam_bs > 0 else 0.0
            
            delays.append(t_opt)
            
            # 记录详细数据
            records.append({
                'method': 'QPSO',
                'step': t,
                'vehicle_id': i,
                'delay_total': t_opt,
                'delay_local': d_loc,
                'delay_edge': d_edge,
                'delay_bs': d_bs,
                'delay_comm_v2v': t_comm_v2v,
                'delay_comm_v2i': t_comm_v2i,
                'k_value': k_val,
                'lambda_local': lam_loc,
                'lambda_edge': lam_edge,
                'lambda_bs': lam_bs,
                'snr_v2v_db': 10 * np.log10(snr_v2v + 1e-15),
                'snr_v2i_db': 10 * np.log10(snr_v2i + 1e-15),
                'A_local': A_l,
                'A_edge': A_e,
                'A_bs': A_b
            })
    
    avg_delay = np.mean(delays)
    return avg_delay, records

def run_comparison_test(n_vehicles, ris_row, ris_col, model_path=None):
    """
    对单个配置运行三种方法的对比测试
    返回: {'ppo': (delay, records), 'ga': (delay, records), 'qpso': (delay, records)}
    """
    results = {}
    
    print(f"\n{'='*80}")
    print(f"Testing Config: Vehicles={n_vehicles}, RIS={ris_row}x{ris_col}")
    print(f"{'='*80}")
    
    # PPO测试
    if model_path and os.path.exists(model_path):
        print("\n[1/3] Running PPO...")
        try:
            avg_delay, records = run_ppo_test(model_path, n_vehicles, ris_row, ris_col)
            results['ppo'] = (avg_delay, records)
            print(f"PPO Average Delay: {avg_delay:.4f} s")
        except Exception as e:
            print(f"PPO test failed: {e}")
            results['ppo'] = (None, None)
    else:
        print("\n[1/3] PPO: Model file not found, skipping...")
        results['ppo'] = (None, None)
    
    # GA测试
    print("\n[2/3] Running GA...")
    try:
        avg_delay, records = run_ga_test(n_vehicles, ris_row, ris_col)
        results['ga'] = (avg_delay, records)
        print(f"GA Average Delay: {avg_delay:.4f} s")
    except Exception as e:
        print(f"GA test failed: {e}")
        results['ga'] = (None, None)
    
    # QPSO测试
    print("\n[3/3] Running QPSO...")
    try:
        avg_delay, records = run_qpso_test(n_vehicles, ris_row, ris_col)
        results['qpso'] = (avg_delay, records)
        print(f"QPSO Average Delay: {avg_delay:.4f} s")
    except Exception as e:
        print(f"QPSO test failed: {e}")
        results['qpso'] = (None, None)
    
    return results

def run_batch_tests():
    """运行批量对比测试"""
    print("="*80)
    print("Batch Comparison Testing: PPO vs GA vs QPSO")
    print("="*80)
    
    # 查找并分类模型文件
    group1, group2 = find_model_files()
    
    print(f"\nGroup 1 (RIS 6x6, different vehicles): {len(group1)} configs")
    for fname, n_veh, r, c in group1:
        print(f"  - {fname}: {n_veh} vehicles, RIS {r}x{c}")
    
    print(f"\nGroup 2 (15 vehicles, different RIS): {len(group2)} configs")
    for fname, n_veh, r, c in group2:
        print(f"  - {fname}: {n_veh} vehicles, RIS {r}x{c}")
    
    # 测试组1
    group1_summary = []  # 汇总数据（用于绘图）
    group1_detailed = []  # 详细数据（用于统计分析）
    print(f"\n{'='*80}")
    print("Testing Group 1: RIS 6x6, Different Vehicle Numbers")
    print(f"{'='*80}")
    for filename, n_veh, ris_r, ris_c in group1:
        results = run_comparison_test(n_veh, ris_r, ris_c, filename)
        
        # 汇总数据
        ppo_delay, ppo_records = results.get('ppo', (None, None))
        ga_delay, ga_records = results.get('ga', (None, None))
        qpso_delay, qpso_records = results.get('qpso', (None, None))
        
        group1_summary.append({
            'filename': filename,
            'n_vehicles': n_veh,
            'ris_row': ris_r,
            'ris_col': ris_c,
            'ppo_delay': ppo_delay,
            'ga_delay': ga_delay,
            'qpso_delay': qpso_delay
        })
        
        # 详细数据（添加配置信息）
        if ppo_records:
            for record in ppo_records:
                record.update({
                    'config_n_vehicles': n_veh,
                    'config_ris_row': ris_r,
                    'config_ris_col': ris_c,
                    'config_filename': filename
                })
                group1_detailed.append(record)
        
        if ga_records:
            for record in ga_records:
                record.update({
                    'config_n_vehicles': n_veh,
                    'config_ris_row': ris_r,
                    'config_ris_col': ris_c,
                    'config_filename': filename
                })
                group1_detailed.append(record)
        
        if qpso_records:
            for record in qpso_records:
                record.update({
                    'config_n_vehicles': n_veh,
                    'config_ris_row': ris_r,
                    'config_ris_col': ris_c,
                    'config_filename': filename
                })
                group1_detailed.append(record)
    
    # 测试组2
    group2_summary = []  # 汇总数据
    group2_detailed = []  # 详细数据
    print(f"\n{'='*80}")
    print("Testing Group 2: 15 Vehicles, Different RIS Sizes")
    print(f"{'='*80}")
    for filename, n_veh, ris_r, ris_c in group2:
        results = run_comparison_test(n_veh, ris_r, ris_c, filename)
        
        # 汇总数据
        ppo_delay, ppo_records = results.get('ppo', (None, None))
        ga_delay, ga_records = results.get('ga', (None, None))
        qpso_delay, qpso_records = results.get('qpso', (None, None))
        
        group2_summary.append({
            'filename': filename,
            'n_vehicles': n_veh,
            'ris_row': ris_r,
            'ris_col': ris_c,
            'ppo_delay': ppo_delay,
            'ga_delay': ga_delay,
            'qpso_delay': qpso_delay
        })
        
        # 详细数据（添加配置信息）
        if ppo_records:
            for record in ppo_records:
                record.update({
                    'config_n_vehicles': n_veh,
                    'config_ris_row': ris_r,
                    'config_ris_col': ris_c,
                    'config_filename': filename
                })
                group2_detailed.append(record)
        
        if ga_records:
            for record in ga_records:
                record.update({
                    'config_n_vehicles': n_veh,
                    'config_ris_row': ris_r,
                    'config_ris_col': ris_c,
                    'config_filename': filename
                })
                group2_detailed.append(record)
        
        if qpso_records:
            for record in qpso_records:
                record.update({
                    'config_n_vehicles': n_veh,
                    'config_ris_row': ris_r,
                    'config_ris_col': ris_c,
                    'config_filename': filename
                })
                group2_detailed.append(record)
    
    # 绘制对比结果
    plot_comparison_results(group1_summary, group2_summary)
    
    # 保存汇总结果
    if group1_summary:
        df1_summary = pd.DataFrame(group1_summary)
        df1_summary.to_csv('comparison_summary_group1_ris6x6.csv', index=False)
        print(f"\nGroup 1 summary saved to 'comparison_summary_group1_ris6x6.csv'")
    
    if group2_summary:
        df2_summary = pd.DataFrame(group2_summary)
        df2_summary.to_csv('comparison_summary_group2_veh15.csv', index=False)
        print(f"Group 2 summary saved to 'comparison_summary_group2_veh15.csv'")
    
    # 保存详细数据
    if group1_detailed:
        df1_detailed = pd.DataFrame(group1_detailed)
        df1_detailed.to_csv('comparison_detailed_group1_ris6x6.csv', index=False)
        print(f"Group 1 detailed data saved to 'comparison_detailed_group1_ris6x6.csv' ({len(df1_detailed)} records)")
    
    if group2_detailed:
        df2_detailed = pd.DataFrame(group2_detailed)
        df2_detailed.to_csv('comparison_detailed_group2_veh15.csv', index=False)
        print(f"Group 2 detailed data saved to 'comparison_detailed_group2_veh15.csv' ({len(df2_detailed)} records)")
    
    return (group1_summary, group1_detailed), (group2_summary, group2_detailed)


def run_rate_power_sweep():
    """
    针对固定配置 (15 车, RIS 6x6)，扫描发射功率：
     扫描发射功率，任务到达率固定为 4 tasks/s
    结果保存到:
      - sweep_summary_veh15_ris6x6.csv
      - sweep_detailed_veh15_ris6x6.csv
    """
    fixed_n_veh = 15
    fixed_ris_row = 6
    fixed_ris_col = 6
    fixed_model = 'ppo_specific_veh15.pth'

    sweep_summary = []
    sweep_detailed = []

    print("\n" + "="*80)
    print("Rate & Power Sweep (Decoupled): 15 Vehicles, RIS 6x6")
    print("="*80)


    #扫描发射功率，任务到达率固定为 4 tasks/s
    fixed_rate = 4.0
    print("\n--- Sweep Tx Power (fixed lambda = 4 tasks/s) ---")
    env_cfg.POISSON_RATE = fixed_rate

    for tx in TX_POWER_LIST:
        env_cfg.TX_POWER_VEHICLE = tx

        print(f"\n[Power Sweep] lambda={fixed_rate} tasks/s, P_tx={tx} W")
        results = run_comparison_test(fixed_n_veh, fixed_ris_row, fixed_ris_col, fixed_model)

        ppo_delay, ppo_records = results.get('ppo', (None, None))
        ga_delay, ga_records = results.get('ga', (None, None))
        qpso_delay, qpso_records = results.get('qpso', (None, None))

        sweep_summary.append({
            'sweep_mode': 'power',
            'n_vehicles': fixed_n_veh,
            'ris_row': fixed_ris_row,
            'ris_col': fixed_ris_col,
            'arrival_rate': fixed_rate,
            'tx_power': tx,
            'ppo_delay': ppo_delay,
            'ga_delay': ga_delay,
            'qpso_delay': qpso_delay
        })

        for method_name, records in [('PPO', ppo_records), ('GA', ga_records), ('QPSO', qpso_records)]:
            if records:
                for rec in records:
                    rec = rec.copy()
                    rec.update({
                        'sweep_mode': 'power',
                        'config_n_vehicles': fixed_n_veh,
                        'config_ris_row': fixed_ris_row,
                        'config_ris_col': fixed_ris_col,
                        'arrival_rate': fixed_rate,
                        'tx_power': tx
                    })
                    sweep_detailed.append(rec)

    # 保存结果
    if sweep_summary:
        df_sum = pd.DataFrame(sweep_summary)
        df_sum.to_csv('sweep_summary_veh15_ris6x6.csv', index=False)
        print("\nSaved 'sweep_summary_veh15_ris6x6.csv'")

    if sweep_detailed:
        df_det = pd.DataFrame(sweep_detailed)
        df_det.to_csv('sweep_detailed_veh15_ris6x6.csv', index=False)
        print(f"Saved 'sweep_detailed_veh15_ris6x6.csv' ({len(df_det)} records)")

def plot_comparison_results(group1_summary, group2_summary):
    """绘制三种方法的对比结果"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 支持负号
    
    # 图1: 组1 - RIS 6x6，不同车辆数
    if group1_summary:
        df1 = pd.DataFrame(group1_summary)
        df1 = df1.sort_values('n_vehicles')
        
        plt.figure(figsize=(12, 6))
        
        # 绘制三条线
        x = df1['n_vehicles'].to_numpy()
        
        if df1['ppo_delay'].notna().any():
            ppo_y = df1['ppo_delay'].to_numpy()
            plt.plot(x, ppo_y, marker='o', linewidth=2, markersize=8, label='PPO', color='b')
        
        if df1['ga_delay'].notna().any():
            ga_y = df1['ga_delay'].to_numpy()
            plt.plot(x, ga_y, marker='s', linewidth=2, markersize=8, label='GA', color='g')
        
        if df1['qpso_delay'].notna().any():
            qpso_y = df1['qpso_delay'].to_numpy()
            plt.plot(x, qpso_y, marker='^', linewidth=2, markersize=8, label='QPSO', color='r')
        
        plt.xlabel('Number of Vehicles', fontsize=12)
        plt.ylabel('Average Delay (s)', fontsize=12)
        plt.title('Method Comparison: Average Delay vs Number of Vehicles (RIS 6×6)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(x)
        
        plt.tight_layout()
        plt.savefig('comparison_group1_ris6x6.png', dpi=200, bbox_inches='tight')
        print("\nSaved 'comparison_group1_ris6x6.png'")
        plt.close()
    
    # 图2: 组2 - 15辆车，不同RIS大小
    if group2_summary:
        df2 = pd.DataFrame(group2_summary)
        df2['ris_size'] = df2['ris_row'].astype(str) + '×' + df2['ris_col'].astype(str)
        df2['ris_elements'] = df2['ris_row'] * df2['ris_col']
        df2 = df2.sort_values('ris_elements')
        
        plt.figure(figsize=(12, 6))
        
        x = df2['ris_elements'].to_numpy()
        x_labels = df2['ris_size'].to_numpy()
        
        if df2['ppo_delay'].notna().any():
            ppo_y = df2['ppo_delay'].to_numpy()
            plt.plot(x, ppo_y, marker='o', linewidth=2, markersize=8, label='PPO', color='b')
        
        if df2['ga_delay'].notna().any():
            ga_y = df2['ga_delay'].to_numpy()
            plt.plot(x, ga_y, marker='s', linewidth=2, markersize=8, label='GA', color='g')
        
        if df2['qpso_delay'].notna().any():
            qpso_y = df2['qpso_delay'].to_numpy()
            plt.plot(x, qpso_y, marker='^', linewidth=2, markersize=8, label='QPSO', color='r')
        
        plt.xlabel('RIS Size (Number of Elements)', fontsize=12)
        plt.ylabel('Average Delay (s)', fontsize=12)
        plt.title('Method Comparison: Average Delay vs RIS Size (15 Vehicles)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(x, x_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig('comparison_group2_veh15.png', dpi=200, bbox_inches='tight')
        print("Saved 'comparison_group2_veh15.png'")
        plt.close()


def plot_analysis_figures():
    """
    使用已保存的CSV文件绘制最终分析图表：
      1) 三种方法：总时延 vs 车辆数（RIS 6x6），上下两个子图（上：PPO，下：GA+QPSO）
      2) 三种方法：总时延 vs RIS大小（15车），同样布局
      3) 三种方法：总时延 vs 发射功率（15车,RIS6x6，sweep_mode='power'）同样布局
      4) 通信时延 (V2V/V2I) vs 发射功率（sweep_detailed）
      5) 箱型图：不同车辆数下三种方法总时延分布（comparison_detailed_group1）
      6) 箱型图：不同RIS尺寸下三种方法总时延分布（comparison_detailed_group2）
    """
    # 设置字体为Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    # 确保所有文本使用Times New Roman
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学文本也使用serif字体

    # ---------- 1) 总时延 vs 车辆数（RIS 6x6） ----------
    if os.path.exists('comparison_summary_group1_ris6x6.csv'):
        df = pd.read_csv('comparison_summary_group1_ris6x6.csv')
        df = df.sort_values('n_vehicles')
        x = df['n_vehicles'].to_numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1]})

        # GA + QPSO 子图（顶部）
        ax1.plot(x, df['ga_delay'].to_numpy(), marker='s', color='tab:green', linewidth=2, markersize=7, label='GA')
        ax1.plot(x, df['qpso_delay'].to_numpy(), marker='^', color='tab:red', linewidth=2, markersize=7, label='QPSO')
        ax1.set_ylabel('Avg Delay (s) - GA/QPSO')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # PPO 子图（底部）
        ax2.plot(x, df['ppo_delay'].to_numpy(), marker='o', color='tab:blue', linewidth=2, markersize=7, label='PPO')
        ax2.set_xlabel('Number of Vehicles')
        ax2.set_ylabel('Avg Delay (s) - PPO')
        ax2.grid(True, linestyle='--', alpha=0.7)

        handles = [
            mpatches.Patch(color='tab:blue', label='PPO'),
            mpatches.Patch(color='tab:green', label='GA'),
            mpatches.Patch(color='tab:red', label='QPSO')
        ]
        ax2.legend(handles=handles, loc='upper left', ncol=3)
        fig.suptitle('Total Delay vs Number of Vehicles (RIS 6×6)', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('analysis_delay_vs_vehicles.png', dpi=200)
        plt.close(fig)

    # ---------- 2) 总时延 vs RIS大小（15车） ----------
    if os.path.exists('comparison_summary_group2_veh15.csv'):
        df = pd.read_csv('comparison_summary_group2_veh15.csv')
        df['ris_elements'] = df['ris_row'] * df['ris_col']
        df['ris_label'] = df['ris_row'].astype(str) + '×' + df['ris_col'].astype(str)
        df = df.sort_values('ris_elements')
        x = df['ris_elements'].to_numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1]})

        # GA + QPSO 子图（顶部）
        ax1.plot(x, df['ga_delay'].to_numpy(), marker='s', color='tab:green', linewidth=2, markersize=7, label='GA')
        ax1.plot(x, df['qpso_delay'].to_numpy(), marker='^', color='tab:red', linewidth=2, markersize=7, label='QPSO')
        ax1.set_ylabel('Avg Delay (s) - GA/QPSO')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # PPO 子图（底部）
        ax2.plot(x, df['ppo_delay'].to_numpy(), marker='o', color='tab:blue', linewidth=2, markersize=7, label='PPO')
        ax2.set_xlabel('RIS Size (Number of Elements)')
        ax2.set_ylabel('Avg Delay (s) - PPO')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['ris_label'], rotation=45)

        handles = [
            mpatches.Patch(color='tab:blue', label='PPO'),
            mpatches.Patch(color='tab:green', label='GA'),
            mpatches.Patch(color='tab:red', label='QPSO')
        ]
        ax2.legend(handles=handles, loc='upper right', ncol=3)
        fig.suptitle('Total Delay vs RIS Size (15 Vehicles)', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('analysis_delay_vs_ris.png', dpi=200)
        plt.close(fig)

    # ---------- 3) 总时延 vs 发射功率（15车, RIS6×6, sweep_mode=power） ----------
    if os.path.exists('sweep_summary_veh15_ris6x6.csv'):
        df = pd.read_csv('sweep_summary_veh15_ris6x6.csv')
        df_power = df[df['sweep_mode'] == 'power'].sort_values('tx_power')
        if not df_power.empty:
            x = df_power['tx_power'].to_numpy()

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1]})

            # GA + QPSO 子图（顶部）
            ax1.plot(x, df_power['ga_delay'].to_numpy(), marker='s', color='tab:green', linewidth=2, markersize=7, label='GA')
            ax1.plot(x, df_power['qpso_delay'].to_numpy(), marker='^', color='tab:red', linewidth=2, markersize=7, label='QPSO')
            ax1.set_ylabel('Avg Delay (s) - GA/QPSO')
            ax1.grid(True, linestyle='--', alpha=0.7)

            # PPO 子图（底部）
            ax2.plot(x, df_power['ppo_delay'].to_numpy(), marker='o', color='tab:blue', linewidth=2, markersize=7, label='PPO')
            ax2.set_xlabel('Transmit Power (W)')
            ax2.set_ylabel('Avg Delay (s) - PPO')
            ax2.grid(True, linestyle='--', alpha=0.7)

            handles = [
                mpatches.Patch(color='tab:blue', label='PPO'),
                mpatches.Patch(color='tab:green', label='GA'),
                mpatches.Patch(color='tab:red', label='QPSO')
            ]
            ax2.legend(handles=handles, loc='upper right', ncol=3)
            fig.suptitle('Total Delay vs Transmit Power (15 Vehicles, RIS 6×6)', fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig('analysis_delay_vs_power.png', dpi=200)
            plt.close(fig)

    # ---------- 4a) V2V通信时延 vs 发射功率（sweep_detailed, power 模式） ----------
    if os.path.exists('sweep_detailed_veh15_ris6x6.csv'):
        df_det = pd.read_csv('sweep_detailed_veh15_ris6x6.csv')
        df_p = df_det[df_det['sweep_mode'] == 'power']
        if not df_p.empty and 'delay_comm_v2v' in df_p.columns:
            # 聚合：按 (method, tx_power) 取平均
            g = df_p.groupby(['method', 'tx_power'])[['delay_comm_v2v']].mean().reset_index()
            methods = ['PPO', 'GA', 'QPSO']
            colors = {'PPO': 'tab:blue', 'GA': 'tab:green', 'QPSO': 'tab:red'}

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1]})

            # GA + QPSO 子图（顶部）
            for m in ['GA', 'QPSO']:
                sub = g[g['method'] == m].sort_values('tx_power')
                if not sub.empty:
                    x = sub['tx_power'].to_numpy()
                    marker = 's' if m == 'GA' else '^'
                    ax1.plot(x, sub['delay_comm_v2v'].to_numpy(), marker=marker, linewidth=2, markersize=6,
                             label=m, color=colors[m])
            ax1.set_ylabel('Comm Delay V2V (s) - GA/QPSO')
            ax1.grid(True, linestyle='--', alpha=0.7)

            # PPO 子图（底部）
            sub_ppo = g[g['method'] == 'PPO'].sort_values('tx_power')
            if not sub_ppo.empty:
                x = sub_ppo['tx_power'].to_numpy()
                ax2.plot(x, sub_ppo['delay_comm_v2v'].to_numpy(), marker='o', linewidth=2, markersize=6,
                         label='PPO', color=colors['PPO'])
            ax2.set_xlabel('Transmit Power (W)')
            ax2.set_ylabel('Comm Delay V2V (s) - PPO')
            ax2.grid(True, linestyle='--', alpha=0.7)

            handles = [
                mpatches.Patch(color='tab:blue', label='PPO'),
                mpatches.Patch(color='tab:green', label='GA'),
                mpatches.Patch(color='tab:red', label='QPSO')
            ]
            ax2.legend(handles=handles, loc='upper right', ncol=3)
            fig.suptitle('V2V Communication Delay vs Transmit Power (15 Vehicles, RIS 6×6)', fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig('analysis_comm_delay_v2v_vs_power.png', dpi=200)
            plt.close(fig)

    # ---------- 4b) V2I通信时延 vs 发射功率（sweep_detailed, power 模式） ----------
    if os.path.exists('sweep_detailed_veh15_ris6x6.csv'):
        df_det = pd.read_csv('sweep_detailed_veh15_ris6x6.csv')
        df_p = df_det[df_det['sweep_mode'] == 'power']
        if not df_p.empty and 'delay_comm_v2i' in df_p.columns:
            # 聚合：按 (method, tx_power) 取平均
            g = df_p.groupby(['method', 'tx_power'])[['delay_comm_v2i']].mean().reset_index()
            methods = ['PPO', 'GA', 'QPSO']
            colors = {'PPO': 'tab:blue', 'GA': 'tab:green', 'QPSO': 'tab:red'}

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1]})

            # GA + QPSO 子图（顶部）
            for m in ['GA', 'QPSO']:
                sub = g[g['method'] == m].sort_values('tx_power')
                if not sub.empty:
                    x = sub['tx_power'].to_numpy()
                    marker = 's' if m == 'GA' else '^'
                    ax1.plot(x, sub['delay_comm_v2i'].to_numpy(), marker=marker, linewidth=2, markersize=6,
                             label=m, color=colors[m])
            ax1.set_ylabel('Comm Delay V2I (s) - GA/QPSO')
            ax1.grid(True, linestyle='--', alpha=0.7)

            # PPO 子图（底部）
            sub_ppo = g[g['method'] == 'PPO'].sort_values('tx_power')
            if not sub_ppo.empty:
                x = sub_ppo['tx_power'].to_numpy()
                ax2.plot(x, sub_ppo['delay_comm_v2i'].to_numpy(), marker='o', linewidth=2, markersize=6,
                         label='PPO', color=colors['PPO'])
            ax2.set_xlabel('Transmit Power (W)')
            ax2.set_ylabel('Comm Delay V2I (s) - PPO')
            ax2.grid(True, linestyle='--', alpha=0.7)

            handles = [
                mpatches.Patch(color='tab:blue', label='PPO'),
                mpatches.Patch(color='tab:green', label='GA'),
                mpatches.Patch(color='tab:red', label='QPSO')
            ]
            ax2.legend(handles=handles, loc='upper right', ncol=3)
            fig.suptitle('V2I Communication Delay vs Transmit Power (15 Vehicles, RIS 6×6)', fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig('analysis_comm_delay_v2i_vs_power.png', dpi=200)
            plt.close(fig)

    # ---------- 5) 箱型图：不同车辆数下三种方法总时延分布 ----------
    if os.path.exists('comparison_detailed_group1_ris6x6.csv'):
        df = pd.read_csv('comparison_detailed_group1_ris6x6.csv')
        methods = ['PPO', 'GA', 'QPSO']
        colors = {'PPO': 'tab:blue', 'GA': 'tab:green', 'QPSO': 'tab:red'}
        veh_list = sorted(df['config_n_vehicles'].unique())
        x_base = np.arange(len(veh_list))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        for j, m in enumerate(methods):
            for i, v in enumerate(veh_list):
                subset = df[(df['config_n_vehicles'] == v) & (df['method'] == m)]['delay_total']
                if len(subset) > 0:
                    # 转换为列表再转为numpy数组，确保是1D数组
                    values_list = subset.tolist()
                    arr = np.array(values_list, dtype=float)
                    # 确保是1D数组
                    if arr.ndim > 1:
                        arr = arr.flatten()
                    if len(arr) > 0:
                        pos = x_base[i] + (j - 1) * width
                        # 逐个绘制箱型图
                        bp = ax.boxplot([arr], positions=[pos], widths=width, patch_artist=True,
                                        boxprops=dict(facecolor=colors[m], alpha=0.6),
                                        medianprops=dict(color='k'))
                        # 设置箱子的颜色
                        for patch in bp['boxes']:
                            patch.set_facecolor(colors[m])
                            patch.set_alpha(0.6)

        ax.set_xticks(x_base)
        ax.set_xticklabels([str(v) for v in veh_list])
        ax.set_xlabel('Number of Vehicles')
        ax.set_ylabel('Total Delay (s)')
        ax.set_title('Delay Distribution vs Number of Vehicles (RIS 6×6)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        handles = [mpatches.Patch(color=colors[m], label=m) for m in methods]
        ax.legend(handles=handles, title='Method')
        plt.tight_layout()
        plt.savefig('boxplot_delay_vs_vehicles.png', dpi=200)
        plt.close(fig)

    # ---------- 6) 箱型图：不同RIS尺寸下三种方法总时延分布 ----------
    if os.path.exists('comparison_detailed_group2_veh15.csv'):
        df = pd.read_csv('comparison_detailed_group2_veh15.csv')
        methods = ['PPO', 'GA', 'QPSO']
        colors = {'PPO': 'tab:blue', 'GA': 'tab:green', 'QPSO': 'tab:red'}
        df['ris_label'] = df['config_ris_row'].astype(str) + '×' + df['config_ris_col'].astype(str)
        ris_list = sorted(df['ris_label'].unique(), key=lambda x: int(x.split('×')[0]) * int(x.split('×')[1]))
        x_base = np.arange(len(ris_list))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        for j, m in enumerate(methods):
            for i, r in enumerate(ris_list):
                subset = df[(df['ris_label'] == r) & (df['method'] == m)]['delay_total']
                if len(subset) > 0:
                    # 转换为列表再转为numpy数组，确保是1D数组
                    values_list = subset.tolist()
                    arr = np.array(values_list, dtype=float)
                    # 确保是1D数组
                    if arr.ndim > 1:
                        arr = arr.flatten()
                    if len(arr) > 0:
                        pos = x_base[i] + (j - 1) * width
                        # 逐个绘制箱型图
                        bp = ax.boxplot([arr], positions=[pos], widths=width, patch_artist=True,
                                        boxprops=dict(facecolor=colors[m], alpha=0.6),
                                        medianprops=dict(color='k'))
                        # 设置箱子的颜色
                        for patch in bp['boxes']:
                            patch.set_facecolor(colors[m])
                            patch.set_alpha(0.6)

        ax.set_xticks(x_base)
        ax.set_xticklabels(ris_list, rotation=45)
        ax.set_xlabel('RIS Size')
        ax.set_ylabel('Total Delay (s)')
        ax.set_title('Delay Distribution vs RIS Size (15 Vehicles)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        handles = [mpatches.Patch(color=colors[m], label=m) for m in methods]
        ax.legend(handles=handles, title='Method')
        plt.tight_layout()
        plt.savefig('boxplot_delay_vs_ris.png', dpi=200)
        plt.close(fig)

if __name__ == '__main__':
    set_seed(RANDOM_SEED)
    # 1. 运行批量对比与参数扫描（如需重新生成数据时打开）
    # run_batch_tests()
    # run_rate_power_sweep()

    # 2. 绘制图表（基于已生成的CSV）
    plot_analysis_figures()

    print("\n" + "="*80)
    print("All plots generated!")
    print("="*80)
