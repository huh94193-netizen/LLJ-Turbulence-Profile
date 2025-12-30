import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import os
import random
import warnings

# ================= 用户配置区域 =================
# 1. 数据文件路径
TARGET_FILE = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'

# 2. 输出路径
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/single_case_validation'

# 3. 【新】一次性生成多少张图？
NUM_PLOTS = 10  

# 4. 场站拟合参数 (来自 Excel)
STATION_PARAMS = {
    # 风速 (Banta)
    'WS_Alpha': 1.65,   
    'WS_Beta': 0.95,    
    
    # 湍流 (Inv Banta)
    'TI_Base': 0.14,    
    'TI_Dip': 0.05,     
    'TI_Alpha': 1.0,    
    'TI_Beta': 1.0,     
    
    # 风向 (Linear)
    'WD_Slope': 0.04    
}

# 5. 判定标准
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 60
MAX_JET_HEIGHT = 480
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 0. 字体配置 ---
def get_safe_font_prop():
    candidates = ['WenQuanYi Micro Hei', 'Zen Hei', 'Droid Sans Fallback', 'SimHei', 'Microsoft YaHei', 'SimSun']
    for font in candidates:
        try:
            if fm.findfont(font) != fm.findfont('DejaVu Sans'): return font
        except: continue
    return None
SYSTEM_ZH_FONT = get_safe_font_prop()
if SYSTEM_ZH_FONT:
    plt.rcParams['font.sans-serif'] = [SYSTEM_ZH_FONT]
    plt.rcParams['axes.unicode_minus'] = False

# --- 1. 模型公式 ---
def model_ws_banta(z, u_jet, z_jet, alpha, beta):
    z_norm = z / z_jet
    z_norm = np.maximum(z_norm, 1e-6)
    return u_jet * np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))

def model_ti_banta_inv(z, z_jet, base, dip, alpha, beta):
    z_norm = z / z_jet
    z_norm = np.maximum(z_norm, 1e-6)
    shape = np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))
    return base - dip * shape

def model_wd_linear(z, wd_jet, z_jet, slope):
    delta_z = z - z_jet
    return wd_jet + slope * delta_z

def unwrap_deg(degrees):
    rads = np.radians(degrees)
    unwrapped = np.unwrap(rads)
    return np.degrees(unwrapped)

# --- 2. 数据读取工具 ---
def is_clean_col(col_name):
    blacklist = ['最大', '最小', '偏差', '矢量', '标准差', 'Max', 'Min', 'Std', 'Gust']
    for bad in blacklist:
        if bad.lower() in col_name.lower(): return False
    return True

def get_pure_col(columns, height, keyword_list):
    candidates = []
    for c in columns:
        if f'{height}m' not in c: continue
        if not any(k in c for k in keyword_list): continue
        if is_clean_col(c): candidates.append(c)
    if not candidates: return None
    candidates.sort(key=len)
    return candidates[0]

def strict_tab_parse(file_path):
    encodings = ['utf-8', 'gbk', 'gb18030', 'latin-1']
    for enc in encodings:
        try:
            temp = pd.read_csv(file_path, sep='\t', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp.columns): return temp
            temp = pd.read_csv(file_path, sep='\s+', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp.columns): return temp
        except: continue
    return None

# --- 3. 核心逻辑 (批量版) ---
def run_simulation_batch():
    filename = os.path.basename(TARGET_FILE)
    print(f"读取数据: {filename} ...")
    
    df = strict_tab_parse(TARGET_FILE)
    if df is None: return
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    # --- 提取高度层 ---
    clean_ws_cols_temp = [c for c in df.columns if ('水平风速' in c or 'Speed' in c) and is_clean_col(c)]
    raw_heights = []
    for c in clean_ws_cols_temp:
        m = re.search(r'(\d+)m', c)
        if m: raw_heights.append(int(m.group(1)))
    heights = sorted(list(set(raw_heights)))
    print(f"高度层: {heights}")

    # --- 矩阵提取 ---
    n_samples = len(df)
    n_heights = len(heights)
    
    cols_ws = [get_pure_col(df.columns, h, ['水平风速', 'Speed']) for h in heights]
    cols_wd = [get_pure_col(df.columns, h, ['风向', 'Direction']) for h in heights]
    cols_std = []
    for h in heights:
        c = next((x for x in df.columns if f'{h}m' in x and ('偏差' in x or 'Std' in x) and '风向' not in x), None)
        cols_std.append(c)

    mat_ws = np.full((n_samples, n_heights), np.nan)
    mat_wd = np.full((n_samples, n_heights), np.nan)
    mat_ti = np.full((n_samples, n_heights), np.nan)

    for i in range(n_heights):
        if cols_ws[i]: mat_ws[:, i] = pd.to_numeric(df[cols_ws[i]], errors='coerce').values
        if cols_wd[i]: mat_wd[:, i] = pd.to_numeric(df[cols_wd[i]], errors='coerce').values
        if cols_std[i] and cols_ws[i]:
            std_val = pd.to_numeric(df[cols_std[i]], errors='coerce').values
            ws_val = mat_ws[:, i]
            with np.errstate(divide='ignore', invalid='ignore'):
                ti_val = std_val / ws_val
                ti_val[ws_val < 3.0] = np.nan
            mat_ti[:, i] = ti_val

    # --- 筛选 LLJ 个例 ---
    print("正在筛选所有急流个例...")
    valid_indices = []
    
    for i in range(n_samples):
        u = mat_ws[i, :]
        if np.isnan(u).any(): continue
        
        idx_max = np.argmax(u)
        u_jet = u[idx_max]
        z_jet = heights[idx_max]
        
        if not (MIN_JET_HEIGHT <= z_jet <= MAX_JET_HEIGHT): continue
        if idx_max == 0 or idx_max == n_heights - 1: continue
        
        if (u_jet - u[0] >= LLJ_THRESHOLD) and (u_jet - u[-1] >= LLJ_THRESHOLD):
            valid_indices.append(i)
            
    print(f"共找到 {len(valid_indices)} 个符合条件的急流时刻。")
    if not valid_indices: return

    # --- 批量抽取 ---
    # 如果可用样本少于要求数量，就全取
    n_to_plot = min(NUM_PLOTS, len(valid_indices))
    selected_indices = random.sample(valid_indices, n_to_plot)
    
    print(f"\n>>> 即将生成 {n_to_plot} 张验证图 <<<")

    # --- 循环绘图 ---
    for count, idx in enumerate(selected_indices):
        timestamp = df.iloc[idx]['Date/Time']
        print(f"[{count+1}/{n_to_plot}] 处理时刻: {timestamp} (Row {idx})")
        
        obs_z = np.array(heights)
        obs_ws = mat_ws[idx, :]
        obs_wd = mat_wd[idx, :]
        obs_ti = mat_ti[idx, :]
        
        idx_max = np.argmax(obs_ws)
        REAL_Z_JET = obs_z[idx_max]
        REAL_U_JET = obs_ws[idx_max]
        
        # 模拟数据
        sim_z = np.linspace(0, 500, 100)
        sim_ws = model_ws_banta(sim_z, REAL_U_JET, REAL_Z_JET, 
                                STATION_PARAMS['WS_Alpha'], STATION_PARAMS['WS_Beta'])
        
        sim_ti = model_ti_banta_inv(sim_z, REAL_Z_JET, 
                                    STATION_PARAMS['TI_Base'], STATION_PARAMS['TI_Dip'],
                                    STATION_PARAMS['TI_Alpha'], STATION_PARAMS['TI_Beta'])
        
        # 风向处理
        obs_wd_unwrapped = unwrap_deg(obs_wd)
        wd_jet_unwrapped = obs_wd_unwrapped[idx_max]
        sim_wd = model_wd_linear(sim_z, wd_jet_unwrapped, REAL_Z_JET, STATION_PARAMS['WD_Slope'])

        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # WS
        axes[0].plot(obs_ws, obs_z, 'ko', markersize=8, label='Obs')
        axes[0].plot(sim_ws, sim_z, 'r-', lw=3, alpha=0.8, label='Sim')
        axes[0].set_title(f'WS (U_jet={REAL_U_JET:.1f}, Z_jet={REAL_Z_JET})')
        axes[0].set_ylabel('Height [m]')
        axes[0].grid(True, ls='--', alpha=0.5)
        axes[0].legend()
        
        # TI
        axes[1].plot(obs_ti, obs_z, 'ko', markersize=8)
        axes[1].plot(sim_ti, sim_z, 'b-', lw=3, alpha=0.8)
        axes[1].set_title('TI Profile')
        axes[1].grid(True, ls='--', alpha=0.5)
        axes[1].set_xlim(0, 0.4)
        
        # WD
        axes[2].plot(obs_wd_unwrapped, obs_z, 'ko', markersize=8)
        axes[2].plot(sim_wd, sim_z, 'g-', lw=3, alpha=0.8)
        axes[2].set_title('WD Profile')
        axes[2].grid(True, ls='--', alpha=0.5)
        
        safe_time = timestamp.replace(":","-").replace(" ","_").replace("/","-")
        fig.suptitle(f"Case Validation: {safe_time} | Station: 1443#", fontsize=14)
        
        plt.tight_layout()
        out_name = os.path.join(OUTPUT_DIR, f'Case_{count+1:02d}_{safe_time}.png')
        plt.savefig(out_name, dpi=150)
        plt.close(fig) # 关掉画布释放内存

    print(f"\n全部 {n_to_plot} 张图片已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_simulation_batch()