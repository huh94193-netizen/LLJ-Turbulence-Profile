import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
from sklearn.metrics import mean_squared_error

# ================= 配置区域 =================
# 建议选一个数据量大的文件
FILE_PATH = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/model_arena_v13'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 质量控制
MIN_AVAILABILITY = 80.0
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

# --- 1. 定义所有候选模型 ---

# [风速模型]
def model_ws_gauss(z, u_base, u_jet, z_jet, sigma):
    return u_base + u_jet * np.exp(-((z - z_jet)**2) / (2 * sigma**2))

def model_ws_asym_gauss(z, u_base, u_jet, z_jet, sigma_down, sigma_up):
    sigma = np.where(z <= z_jet, sigma_down, sigma_up)
    return u_base + u_jet * np.exp(-((z - z_jet)**2) / (2 * sigma**2))

# [风向模型]
def model_wd_linear(z, a, b):
    return a * z + b

def model_wd_quad(z, a, b, c):
    return a * z**2 + b * z + c

# [湍流模型]
def model_ti_power(z, a, b):
    # TI = a * z^(-b)
    return a * np.power(z, -b)

def model_ti_exp(z, a, b, c):
    # TI = a * exp(-b*z) + c
    return a * np.exp(-b * z) + c

# --- 2. 辅助工具 ---
def strict_tab_parse_v3(file_path):
    # ... (保持原有的解析逻辑，为了节省篇幅，此处省略，请确保包含此函数) ...
    # 这里是一个简化的占位，实际运行请复制之前的 strict_tab_parse_v3 函数
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'latin-1']
    raw_lines = []
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                raw_lines = f.readlines()
            break
        except: continue
    if not raw_lines: return None
    header_idx = -1
    for i, line in enumerate(raw_lines[:100]):
        if "Date/Time" in line or "m水平风速" in line:
            header_idx = i
            break     
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

def unwrap_deg(degrees):
    """处理风向 350->10 的突变，使其变成 350->370，方便线性拟合"""
    rads = np.radians(degrees)
    unwrapped_rads = np.unwrap(rads)
    return np.degrees(unwrapped_rads)

def circular_rmse(true, pred):
    """计算风向的环形误差"""
    diff = np.abs(true - pred) % 360
    diff = np.minimum(diff, 360 - diff)
    return np.sqrt(np.mean(diff**2))

# --- 3. 核心逻辑 ---
def run_arena(file_path):
    print(f"正在读取数据: {os.path.basename(file_path)}")
    df_raw = strict_tab_parse_v3(file_path)
    if df_raw is None: return

    # 清洗与提取
    df_raw.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df_raw.columns]
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c and '偏差' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    # 提取所有数据到内存
    data_dict = {}
    for h in heights:
        data_dict[f'ws_{h}'] = pd.to_numeric(df_raw[f'{h}m水平风速'], errors='coerce')
        # 找风向
        wd_col = next((c for c in df_raw.columns if str(h) in c and '风向' in c and '最大' not in c), None)
        if wd_col: data_dict[f'wd_{h}'] = pd.to_numeric(df_raw[wd_col], errors='coerce')
        # 找TI (偏差/风速)
        std_col = next((c for c in df_raw.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)
        if std_col: 
            std_val = pd.to_numeric(df_raw[std_col], errors='coerce')
            data_dict[f'std_{h}'] = std_val

    df = pd.DataFrame(data_dict)
    
    # 筛选 LLJ
    ws_cols = [f'ws_{h}' for h in heights]
    df_clean = df.dropna(subset=ws_cols)
    
    events_data = [] # 存储待拟合的原始数据
    
    print(" -> 正在筛选 LLJ 事件...")
    for idx in df_clean.index:
        ws_vals = df_clean.loc[idx, ws_cols].values
        
        # LLJ 判断
        try:
            mx_i = np.nanargmax(ws_vals)
            mx_h = heights[mx_i]
            if mx_h <= MIN_JET_HEIGHT or mx_h >= MAX_JET_HEIGHT: continue
            if (ws_vals[mx_i] - ws_vals[0] >= LLJ_THRESHOLD) and (ws_vals[mx_i] - ws_vals[-1] >= LLJ_THRESHOLD):
                pass # 是急流
            else:
                continue
        except: continue
        
        # 提取该时刻的三种廓线
        row_data = {'idx': idx, 'h': np.array(heights), 'ws': ws_vals}
        
        # 提取 WD
        wd_vals = []
        for h in heights:
            k = f'wd_{h}'
            wd_vals.append(df.loc[idx, k] if k in df else np.nan)
        row_data['wd'] = np.array(wd_vals)
        
        # 提取 TI
        ti_vals = []
        for h, w in zip(heights, ws_vals):
            k = f'std_{h}'
            if k in df and w > 3.0: # 风速>3才算TI
                ti_vals.append(df.loc[idx, k] / w)
            else:
                ti_vals.append(np.nan)
        row_data['ti'] = np.array(ti_vals)
        
        events_data.append(row_data)

    print(f" -> 捕获 {len(events_data)} 个事件，开始全维度模型竞技...")

    # 结果容器
    arena_res = {
        'Speed': {'Gaussian': [], 'Asym_Gaussian': []},
        'Dir': {'Linear': [], 'Quadratic': []},
        'TI': {'Power': [], 'Exponential': []}
    }
    
    # 参数记录表 (用于 CSV)
    param_records = []

    # --- 循环拟合 ---
    for i, ev in enumerate(events_data):
        z = ev['h']
        ws = ev['ws']
        wd = ev['wd']
        ti = ev['ti']
        
        rec = {'Event_ID': ev['idx']}
        
        # 1. Speed Arena
        max_u = np.max(ws)
        max_z = z[np.argmax(ws)]
        
        # Model A: Gaussian
        try:
            p0 = [ws[0], max_u-ws[0], max_z, 50]
            popt, _ = curve_fit(model_ws_gauss, z, ws, p0=p0, maxfev=1000)
            rmse = np.sqrt(mean_squared_error(ws, model_ws_gauss(z, *popt)))
            arena_res['Speed']['Gaussian'].append(rmse)
        except: pass
        
        # Model B: Asym Gaussian (Record Params)
        try:
            p0 = [ws[0], max_u-ws[0], max_z, 40, 60]
            popt, _ = curve_fit(model_ws_asym_gauss, z, ws, p0=p0, maxfev=1000)
            rmse = np.sqrt(mean_squared_error(ws, model_ws_asym_gauss(z, *popt)))
            arena_res['Speed']['Asym_Gaussian'].append(rmse)
            
            # 记录参数
            rec['WS_Model'] = 'Asym_Gaussian'
            rec['WS_Ubase'] = popt[0]
            rec['WS_Ujet'] = popt[1]
            rec['WS_Zjet'] = popt[2]
            rec['WS_SigmaD'] = popt[3]
            rec['WS_SigmaU'] = popt[4]
            rec['WS_RMSE'] = rmse
        except: 
            rec['WS_RMSE'] = np.nan

        # 2. Direction Arena
        # 去除 NaN 并处理 wrap
        mask_wd = ~np.isnan(wd)
        if np.sum(mask_wd) > 3:
            z_wd = z[mask_wd]
            wd_clean = unwrap_deg(wd[mask_wd]) # 关键：解缠绕
            
            # Model A: Linear
            try:
                popt, _ = curve_fit(model_wd_linear, z_wd, wd_clean, maxfev=1000)
                pred = model_wd_linear(z_wd, *popt) % 360
                rmse = circular_rmse(wd[mask_wd], pred)
                arena_res['Dir']['Linear'].append(rmse)
            except: pass
            
            # Model B: Quadratic (Record Params)
            try:
                popt, _ = curve_fit(model_wd_quad, z_wd, wd_clean, maxfev=1000)
                pred = model_wd_quad(z_wd, *popt) % 360
                rmse = circular_rmse(wd[mask_wd], pred)
                arena_res['Dir']['Quadratic'].append(rmse)
                
                rec['WD_Model'] = 'Quadratic'
                rec['WD_a'] = popt[0]
                rec['WD_b'] = popt[1]
                rec['WD_c'] = popt[2]
                rec['WD_RMSE'] = rmse
            except: 
                rec['WD_RMSE'] = np.nan
        
        # 3. TI Arena
        mask_ti = ~np.isnan(ti)
        if np.sum(mask_ti) > 3:
            z_ti = z[mask_ti]
            ti_clean = ti[mask_ti]
            
            # Model A: Power
            try:
                popt, _ = curve_fit(model_ti_power, z_ti, ti_clean, p0=[0.1, 0.1], maxfev=1000)
                rmse = np.sqrt(mean_squared_error(ti_clean, model_ti_power(z_ti, *popt)))
                arena_res['TI']['Power'].append(rmse)
                
                # 记录 Power Law 参数 (因为它通常更通用)
                rec['TI_Model'] = 'PowerLaw'
                rec['TI_a'] = popt[0]
                rec['TI_b'] = popt[1] # Decay exponent
                rec['TI_RMSE'] = rmse
            except: pass
            
            # Model B: Exponential
            try:
                popt, _ = curve_fit(model_ti_exp, z_ti, ti_clean, p0=[0.2, 0.005, 0.05], maxfev=1000)
                rmse = np.sqrt(mean_squared_error(ti_clean, model_ti_exp(z_ti, *popt)))
                arena_res['TI']['Exponential'].append(rmse)
            except: pass

        param_records.append(rec)

    # --- 绘图与保存 ---
    
    # 1. 保存参数 CSV
    df_params = pd.DataFrame(param_records)
    csv_path = os.path.join(OUTPUT_DIR, 'LLJ_Fitted_Parameters.csv')
    df_params.to_csv(csv_path, index=False)
    print(f"\n[数据] 拟合参数表已保存: {csv_path}")
    print("      (包含每个事件的 Asym_Gaussian, Quadratic Dir, Power TI 参数)")

    # 2. 绘制 Boxplot
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Speed Plot
    data_s = [arena_res['Speed']['Gaussian'], arena_res['Speed']['Asym_Gaussian']]
    axes[0].boxplot(data_s, labels=['Gaussian\n(Sym)', 'Asym\nGaussian'], patch_artist=True, boxprops=dict(facecolor='#1f77b4'))
    axes[0].set_title('Wind Speed Model Error (RMSE)')
    axes[0].set_ylabel('RMSE [m/s]')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylim(0, 1.5) # 限制Y轴以便看清中位数

    # Dir Plot
    data_d = [arena_res['Dir']['Linear'], arena_res['Dir']['Quadratic']]
    axes[1].boxplot(data_d, labels=['Linear', 'Quadratic'], patch_artist=True, boxprops=dict(facecolor='purple'))
    axes[1].set_title('Wind Dir Model Error (RMSE)')
    axes[1].set_ylabel('RMSE [deg]')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_ylim(0, 20)

    # TI Plot
    data_t = [arena_res['TI']['Power'], arena_res['TI']['Exponential']]
    axes[2].boxplot(data_t, labels=['Power Law', 'Exponential'], patch_artist=True, boxprops=dict(facecolor='#d62728'))
    axes[2].set_title('TI Model Error (RMSE)')
    axes[2].set_ylabel('RMSE [-]')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_ylim(0, 0.05)
    
    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, 'Model_Arena_Results.png')
    plt.savefig(img_path, dpi=300)
    print(f"[图表] 竞技场对比图已保存: {img_path}")

if __name__ == "__main__":
    run_arena(FILE_PATH)