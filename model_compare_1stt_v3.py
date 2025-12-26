import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import re
import os

# ================= 配置区域 =================
# 替换为你的文件路径
FILE_PATH = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/global_optimization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 质量控制
MIN_AVAILABILITY = 80.0
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

# --- 1. 数据解析与矩阵构建 ---
def load_and_extract_matrices(file_path):
    print(f"正在读取并矩阵化数据: {os.path.basename(file_path)}")
    
    # A. 基础读取 (简化版)
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'latin-1']
    raw_lines = []
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f: raw_lines = f.readlines()
            break
        except: continue
    if not raw_lines: return None, None, None
    
    header_idx = -1
    for i, line in enumerate(raw_lines[:100]):
        if "m水平风速" in line:
            header_idx = i; break
    if header_idx == -1: return None, None, None
    
    header = raw_lines[header_idx].strip().split('\t')
    header = [h.strip().replace('"', '') for h in header]
    data = [line.strip().split('\t') for line in raw_lines[header_idx+1:] if line.strip()]
    # 补齐
    max_len = len(header)
    data = [d[:max_len] + ['']*(max_len-len(d)) for d in data]
    df = pd.DataFrame(data, columns=header)
    
    # B. 提取高度列
    df.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df.columns]
    speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    z_vec = np.array(heights)
    
    # C. 构建大矩阵 (Events x Heights)
    # 提取所有数据到 numpy 数组，极大加速计算
    n_samples = len(df)
    n_heights = len(heights)
    
    ws_matrix = np.full((n_samples, n_heights), np.nan)
    wd_matrix = np.full((n_samples, n_heights), np.nan)
    ti_matrix = np.full((n_samples, n_heights), np.nan)
    
    for i, h in enumerate(heights):
        ws_col = f'{h}m水平风速'
        if ws_col in df: ws_matrix[:, i] = pd.to_numeric(df[ws_col], errors='coerce')
        
        wd_col = next((c for c in df.columns if str(h) in c and '风向' in c and '最大' not in c), None)
        if wd_col: wd_matrix[:, i] = pd.to_numeric(df[wd_col], errors='coerce')
        
        std_col = next((c for c in df.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)
        if std_col:
            std = pd.to_numeric(df[std_col], errors='coerce')
            ws = ws_matrix[:, i]
            ti = std / ws
            ti[ws < 3.0] = np.nan
            ti_matrix[:, i] = ti

    # D. 筛选 LLJ 事件并准备优化所需的输入向量
    valid_events = []
    
    # 预先分配列表
    list_ws_obs = []
    list_wd_obs = []
    list_ti_obs = []
    
    # 存储每个事件的“环境变量” (Variable Parameters)
    list_params = [] # 存 [u_base, u_jet, z_jet_val, idx_z_jet]
    
    for i in range(n_samples):
        s = ws_matrix[i, :]
        if np.isnan(s).any(): continue # 简单起见，丢弃含缺测的行
        
        mx_i = np.argmax(s)
        mx_v = s[mx_i]
        mx_h = heights[mx_i]
        
        # LLJ 判断
        if mx_h <= MIN_JET_HEIGHT or mx_h >= MAX_JET_HEIGHT: continue
        if (mx_v - s[0] >= LLJ_THRESHOLD) and (mx_v - s[-1] >= LLJ_THRESHOLD):
            # 这是一个 LLJ 事件
            u_base = s[0]
            u_jet = mx_v - u_base
            z_jet = mx_h
            
            list_ws_obs.append(s)
            list_wd_obs.append(wd_matrix[i, :])
            list_ti_obs.append(ti_matrix[i, :])
            list_params.append([u_base, u_jet, z_jet])
            
    if not list_ws_obs: return None, None, None
    
    print(f" -> 筛选出 {len(list_ws_obs)} 个高质量 LLJ 事件用于全局训练")
    
    return (
        z_vec, 
        np.array(list_ws_obs), 
        np.array(list_wd_obs), 
        np.array(list_ti_obs), 
        np.array(list_params) # [N, 3] -> col0: u_base, col1: u_jet, col2: z_jet
    )

# --- 2. 优化求解器 (The Solver) ---

def unwrap_matrix(wd_mat):
    """批量解缠绕风向"""
    rads = np.radians(wd_mat)
    unwrapped = np.unwrap(rads, axis=1)
    return np.degrees(unwrapped)

def solve_global_parameters(z_vec, ws_obs, wd_obs, ti_obs, event_params):
    """
    核心函数：寻找使得所有事件总误差最小的全局参数
    event_params: [u_base, u_jet, z_jet] for each event
    """
    n_events = len(ws_obs)
    
    # ================= 赛道 1: 风速 (Asymmetric Gaussian) =================
    print("\n[计算中] 正在寻找最优风速模型参数 (Sigma_Down, Sigma_Up)...")
    
    # 提取环境变量
    U_base = event_params[:, 0][:, np.newaxis] # Shape (N, 1)
    U_jet  = event_params[:, 1][:, np.newaxis]
    Z_jet  = event_params[:, 2][:, np.newaxis]
    Z_grid = z_vec[np.newaxis, :]            # Shape (1, M)
    
    def ws_loss_func(params):
        sigma_d, sigma_u = params
        # 向量化计算模型值
        # sigma 根据高度选择: 如果 z <= z_jet 用 sigma_d, 否则 sigma_u
        sigma_grid = np.where(Z_grid <= Z_jet, sigma_d, sigma_u)
        
        # 公式: u = u_base + u_jet * exp(...)
        term = -((Z_grid - Z_jet)**2) / (2 * sigma_grid**2)
        ws_pred = U_base + U_jet * np.exp(term)
        
        # 计算 RMSE
        error = ws_obs - ws_pred
        return np.sum(error**2) # 最小化残差平方和

    # 初始猜测 [40, 60]
    res_ws = minimize(ws_loss_func, [40.0, 60.0], bounds=[(10, 200), (10, 200)], method='L-BFGS-B')
    best_sigma_d, best_sigma_u = res_ws.x
    
    # ================= 赛道 2: 湍流 (Power Law) =================
    print("[计算中] 正在寻找最优湍流衰减指数 (Alpha)...")
    # 模型: TI(z) = TI_ref * (z / z_ref)^(-alpha)
    # 这里的变量是: TI_ref (每个事件的底层TI)
    # 全局参数是: alpha
    
    # 提取底层 TI 作为 TI_ref
    valid_ti_mask = ~np.isnan(ti_obs).any(axis=1)
    ti_clean = ti_obs[valid_ti_mask]
    
    if len(ti_clean) > 0:
        TI_ref = ti_clean[:, 0][:, np.newaxis] # 取第一个高度的TI
        Z_ref = z_vec[0]
        
        def ti_loss_func(params):
            alpha = params[0]
            ti_pred = TI_ref * np.power(Z_grid / Z_ref, -alpha)
            error = ti_clean - ti_pred
            return np.sum(error**2)
            
        res_ti = minimize(ti_loss_func, [0.5], bounds=[(0.01, 2.0)])
        best_ti_alpha = res_ti.x[0]
    else:
        best_ti_alpha = np.nan

    # ================= 赛道 3: 风向 (Linear Veering) =================
    print("[计算中] 正在寻找最优风向偏转率 (Veering Rate)...")
    # 模型: WD(z) = WD_base + rate * (z - z_base) + curvature * (z-z_base)^2
    # 变量: WD_base (底层风向)
    # 全局参数: rate (线性变化率), curvature (二次项系数)
    
    valid_wd_mask = ~np.isnan(wd_obs).any(axis=1)
    wd_clean = wd_obs[valid_wd_mask]
    
    if len(wd_clean) > 0:
        wd_unwrapped = unwrap_matrix(wd_clean)
        WD_base = wd_unwrapped[:, 0][:, np.newaxis]
        Z_base = z_vec[0]
        
        def wd_loss_func(params):
            rate, curve = params
            delta_z = Z_grid - Z_base
            wd_pred = WD_base + rate * delta_z + curve * (delta_z**2)
            error = wd_unwrapped - wd_pred
            return np.sum(error**2)
            
        res_wd = minimize(wd_loss_func, [0.1, 0.0], method='L-BFGS-B')
        best_wd_rate, best_wd_curve = res_wd.x
    else:
        best_wd_rate, best_wd_curve = np.nan, np.nan

    return (best_sigma_d, best_sigma_u), best_ti_alpha, (best_wd_rate, best_wd_curve)

# --- 3. 输出报告 ---
def main():
    z_vec, ws_obs, wd_obs, ti_obs, event_params = load_and_extract_matrices(FILE_PATH)
    if z_vec is None: return
    
    # 运行全局优化
    (sig_d, sig_u), ti_alpha, (wd_rate, wd_curve) = solve_global_parameters(z_vec, ws_obs, wd_obs, ti_obs, event_params)
    
    print("\n" + "="*70)
    print(" >>> 场站 LLJ 通用拟合公式 (Global Best-Fit Formulas) <<<")
    print("="*70)
    print("说明: 以下公式中的系数是针对该场站所有急流事件优化的'最佳固定参数'。")
    print("      实际使用时，只需输入当下的 Z_jet, U_jet, U_base 即可还原廓线。")
    print("-" * 70)
    
    # 1. 风速公式
    print(f"\n【1】风速廓线模型 (Asymmetric Gaussian)")
    print(f"     最佳下层厚度 (Sigma_Down): {sig_d:.2f}")
    print(f"     最佳上层厚度 (Sigma_Up)  : {sig_u:.2f}")
    print(f"\n     >> 通用公式:")
    print(f"     U(z) = U_base + U_jet * exp( -((z - Z_jet)^2) / (2 * Sigma^2) )")
    print(f"         其中 Sigma = {sig_d:.2f} (当 z <= Z_jet)")
    print(f"                      {sig_u:.2f} (当 z >  Z_jet)")
    
    # 2. 湍流公式
    print(f"\n【2】湍流廓线模型 (Power Law Decay)")
    print(f"     最佳衰减指数 (Alpha): {ti_alpha:.4f}")
    print(f"\n     >> 通用公式:")
    print(f"     TI(z) = TI_ref * (z / {z_vec[0]})^(-{ti_alpha:.4f})")
    print(f"         (TI_ref 为最低高度 {z_vec[0]}m 处的实测湍流)")

    # 3. 风向公式
    print(f"\n【3】风向廓线模型 (Quadratic Veering)")
    print(f"     最佳线性变化率: {wd_rate:.4f} deg/m")
    print(f"     最佳曲率系数  : {wd_curve:.2e} deg/m^2")
    print(f"\n     >> 通用公式:")
    sign_c = "+" if wd_curve >= 0 else ""
    sign_r = "+" if wd_rate >= 0 else ""
    print(f"     WD(z) = WD_base {sign_r} {wd_rate:.4f}*(z - {z_vec[0]}) {sign_c} {wd_curve:.2e}*(z - {z_vec[0]})^2")
    print(f"         (WD_base 为最低高度 {z_vec[0]}m 处的实测风向)")

    # 保存到文本
    with open(os.path.join(OUTPUT_DIR, 'Site_Universal_Formulas.txt'), 'w') as f:
        f.write("Field Station Universal LLJ Formulas\n")
        f.write("Optimization Method: Global Residual Minimization over all LLJ events\n\n")
        f.write(f"1. Wind Speed (Asym Gaussian):\n   Sigma_Down={sig_d:.4f}, Sigma_Up={sig_u:.4f}\n\n")
        f.write(f"2. TI (Power Law):\n   Alpha={ti_alpha:.4f}, Z_ref={z_vec[0]}\n\n")
        f.write(f"3. Wind Dir (Quadratic):\n   Rate={wd_rate:.5f}, Curve={wd_curve:.5e}, Z_ref={z_vec[0]}\n")
    
    print(f"\n[完成] 结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()