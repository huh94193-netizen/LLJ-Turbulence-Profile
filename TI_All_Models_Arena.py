import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import re
import os
import warnings

# ================= 配置区域 =================
# 替换为您的数据文件路径
FILE_PATH = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/ti_model_arena'

# 质量控制
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 1. 定义模型群 (The Models) ---

def model_inv_gaussian(z, ti_base, ti_dip, z_jet, sigma):
    """
    [模型 1] 倒置高斯 (对称)
    """
    return ti_base - ti_dip * np.exp(-((z - z_jet)**2) / (2 * sigma**2))

def model_inv_asym_gaussian(z, ti_base, ti_dip, z_jet, sigma_down, sigma_up):
    """
    [模型 2] 倒置非对称高斯
    """
    sigma = np.where(z <= z_jet, sigma_down, sigma_up)
    return ti_base - ti_dip * np.exp(-((z - z_jet)**2) / (2 * sigma**2))

def model_inv_banta(z, ti_base, ti_dip, z_jet, alpha, beta):
    """
    [模型 3] 倒置 Banta (Wall Jet Dip) - 您的创新点
    TI = Base - Dip * Banta_Shape
    """
    z_norm = z / z_jet
    z_norm = np.maximum(z_norm, 1e-6) # 保护
    # Banta 形状函数 (峰值为 1)
    shape = np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))
    return ti_base - ti_dip * shape

# --- 2. 数据读取与处理 ---
def strict_tab_parse_v3(file_path):
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16', 'latin-1']
    raw_lines = []
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f: raw_lines = f.readlines()
            break
        except: continue
    if not raw_lines: return None
    
    header_idx = -1
    for i, line in enumerate(raw_lines[:100]):
        if "m水平风速" in line: header_idx = i; break
    if header_idx == -1: return None
    
    header = raw_lines[header_idx].strip().split('\t')
    header = [h.strip().replace('"', '') for h in header]
    data = []
    for i in range(header_idx + 1, len(raw_lines)):
        line = raw_lines[i].strip()
        if not line: continue
        parts = line.split('\t')
        parts = [p.strip().replace('"', '') for p in parts]
        if len(parts) > len(header): parts = parts[:len(header)]
        elif len(parts) < len(header): parts += [''] * (len(header) - len(parts))
        data.append(parts)
    return pd.DataFrame(data, columns=header)

def get_mean_ti_profile(file_path):
    print(f"正在读取并提取 TI 廓线: {os.path.basename(file_path)}")
    df_raw = strict_tab_parse_v3(file_path)
    if df_raw is None: return None, None

    df_raw.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df_raw.columns]
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    # 提取矩阵
    n = len(df_raw)
    ws_mat = np.full((n, len(heights)), np.nan)
    ti_mat = np.full((n, len(heights)), np.nan)
    
    for i, h in enumerate(heights):
        ws_c = f'{h}m水平风速'
        # 模糊匹配 TI 列 (偏差/标准差)
        std_c = next((c for c in df_raw.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)
        
        if ws_c in df_raw and std_c:
            w = pd.to_numeric(df_raw[ws_c], errors='coerce').values
            s = pd.to_numeric(df_raw[std_c], errors='coerce').values
            ws_mat[:, i] = w
            with np.errstate(divide='ignore', invalid='ignore'):
                ti = s / w
                ti[w < 3.0] = np.nan
            ti_mat[:, i] = ti
            
    # 筛选 LLJ 事件
    llj_ti_list = []
    
    for i in range(n):
        u = ws_mat[i, :]
        if np.isnan(u).any(): continue
        mx_i = np.argmax(u)
        z_jet = heights[mx_i]
        
        if (z_jet > MIN_JET_HEIGHT) and (z_jet < MAX_JET_HEIGHT):
            if (u[mx_i] - u[0] >= LLJ_THRESHOLD) and (u[mx_i] - u[-1] >= LLJ_THRESHOLD):
                if not np.isnan(ti_mat[i, :]).any():
                    llj_ti_list.append(ti_mat[i, :])
                    
    if len(llj_ti_list) < 10:
        print("急流样本不足。")
        return None, None
        
    print(f" -> 基于 {len(llj_ti_list)} 个急流事件计算平均 TI 廓线...")
    mean_ti = np.mean(llj_ti_list, axis=0)
    return np.array(heights), mean_ti

# --- 3. 竞技场主逻辑 ---
def run_ti_arena(file_path):
    z_vals, ti_vals = get_mean_ti_profile(file_path)
    if z_vals is None: return

    # 准备拟合
    results = {}
    z_smooth = np.linspace(0, 500, 200)
    
    # 初值猜测
    ti_min = np.min(ti_vals)
    ti_max = np.max(ti_vals)
    z_at_min = z_vals[np.argmin(ti_vals)] # 假设凹陷最深处就是 jet height
    
    # 1. Inverted Gaussian
    try:
        # p0: base, dip, z_jet, sigma
        p0 = [ti_max, ti_max - ti_min, z_at_min, 50]
        popt, _ = curve_fit(model_inv_gaussian, z_vals, ti_vals, p0=p0, maxfev=2000)
        rmse = np.sqrt(mean_squared_error(ti_vals, model_inv_gaussian(z_vals, *popt)))
        results['Inv Gaussian'] = {'rmse': rmse, 'func': model_inv_gaussian, 'popt': popt, 'color': 'blue', 'ls': '--'}
    except: pass
    
    # 2. Inverted Asym Gaussian
    try:
        # p0: base, dip, z_jet, sig_d, sig_u
        p0 = [ti_max, ti_max - ti_min, z_at_min, 40, 60]
        popt, _ = curve_fit(model_inv_asym_gaussian, z_vals, ti_vals, p0=p0, maxfev=5000)
        rmse = np.sqrt(mean_squared_error(ti_vals, model_inv_asym_gaussian(z_vals, *popt)))
        results['Inv Asym Gaussian'] = {'rmse': rmse, 'func': model_inv_asym_gaussian, 'popt': popt, 'color': 'green', 'ls': '-.'}
    except: pass
    
    # 3. Inverted Banta (The Challenger)
    try:
        # p0: base, dip, z_jet, alpha, beta
        # alpha, beta 初始设为 1.0 (线性衰减) 到 2.0 (抛物线)
        p0 = [ti_max, ti_max - ti_min, z_at_min, 1.0, 1.0]
        # 约束 alpha, beta > 0
        bounds = ([0, 0, 50, 0.1, 0.1], [1, 1, 500, 10, 10])
        popt, _ = curve_fit(model_inv_banta, z_vals, ti_vals, p0=p0, bounds=bounds, maxfev=5000)
        rmse = np.sqrt(mean_squared_error(ti_vals, model_inv_banta(z_vals, *popt)))
        results['Inv Banta'] = {'rmse': rmse, 'func': model_inv_banta, 'popt': popt, 'color': 'red', 'ls': '-'}
    except Exception as e: 
        print(f"Banta Fit Failed: {e}")

    # --- 输出与绘图 ---
    print("\n" + "="*50)
    print(" 🏆 湍流模型竞技场 (TI Model Arena)")
    print("="*50)
    
    sorted_res = sorted(results.items(), key=lambda x: x[1]['rmse'])
    for name, res in sorted_res:
        print(f"{name:20s} | RMSE: {res['rmse']:.5f}")
        if name == 'Inv Banta':
            p = res['popt']
            print(f"   -> Params: Alpha={p[3]:.2f}, Beta={p[4]:.2f} (Shape Factor)")

    best_model = sorted_res[0][0]
    print(f"\n>>> 胜出者: 【{best_model}】")
    
    # 绘图
    plt.figure(figsize=(9, 8))
    plt.plot(ti_vals, z_vals, 'ko', markersize=8, label='Observed Mean TI', zorder=10)
    
    for name, res in reversed(sorted_res): # 越好的后画
        lw = 3 if name == best_model else 1.5
        alpha = 1.0 if name == best_model else 0.7
        y_pred = res['func'](z_smooth, *res['popt'])
        plt.plot(y_pred, z_smooth, color=res['color'], linestyle=res['ls'], linewidth=lw, alpha=alpha, 
                 label=f'{name} (RMSE={res["rmse"]:.4f})')
        
    plt.title(f'TI Profile Model Comparison\nIs TI shape "Banta-like"?', fontsize=14)
    plt.xlabel('Turbulence Intensity [-]', fontsize=12)
    plt.ylabel('Height [m]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    out_img = os.path.join(OUTPUT_DIR, 'TI_Banta_Vs_Gaussian.png')
    plt.savefig(out_img, dpi=300)
    print(f"\n[图表] 对比图已保存: {out_img}")

if __name__ == "__main__":
    run_ti_arena(FILE_PATH)