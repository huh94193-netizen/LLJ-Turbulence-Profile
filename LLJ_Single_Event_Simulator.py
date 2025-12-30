import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import os
import random
import warnings

# ================= 用户配置区域 =================
# 1. 数据文件路径 (请修改为您想验证的那个场站文件)
TARGET_FILE = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'

# 2. 输出路径
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/single_case_validation'

# 3. 【关键】在此填入该场站拟合好的参数 (来自您的 Excel)
# 下面是示例值，请替换为双鸭山集贤的真实拟合结果！
STATION_PARAMS = {
    # 风速 (Banta Model)
    'WS_Alpha': 0.91,   # 双鸭山集贤，替换为 Excel 中的 WS_Alpha
    'WS_Beta': 0.96,    # 双鸭山集贤，替换为 Excel 中的 WS_Beta
    
    # 湍流 (Inv Banta Model)
    'TI_Base': 0.4,    # 双鸭山集贤，替换为 Excel 中的 TI_Base
    'TI_Dip': 0.34,     # 双鸭山集贤，替换为 Excel 中的 TI_Dip
    'TI_Alpha': 0.16,    # 双鸭山集贤，替换为 Excel 中的 TI_Alpha
    'TI_Beta': 0.14,     # 双鸭山集贤，替换为 Excel 中的 TI_Beta
    
    # 风向 (Linear Model)
    'WD_Slope': 0.086    # 双鸭山集贤，替换为 Excel 中的 WD_Linear_k
}

# 4. 判定标准 (保持一致)
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

# --- 1. 模型公式 (用于生成模拟曲线) ---

def model_ws_banta(z, u_jet, z_jet, alpha, beta):
    """
    生成风速模拟曲线
    输入绝对高度 z，利用通用参数 alpha/beta 和个例的 u_jet/z_jet 还原
    """
    z_norm = z / z_jet
    z_norm = np.maximum(z_norm, 1e-6)
    # Banta Norm 公式: (z/Z)^alpha * exp(beta * (1-z/Z))
    # 还原绝对风速: * U_jet
    return u_jet * np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))

def model_ti_banta_inv(z, z_jet, base, dip, alpha, beta):
    """生成湍流模拟曲线"""
    z_norm = z / z_jet
    z_norm = np.maximum(z_norm, 1e-6)
    shape = np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))
    return base - dip * shape

def model_wd_linear(z, wd_jet, z_jet, slope):
    """
    生成风向模拟曲线 (相对转角模型)
    WD(z) = WD_jet + k * (z - z_jet)
    """
    delta_z = z - z_jet
    return wd_jet + slope * delta_z

# --- 2. 数据读取工具 (复用稳健版) ---
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

def unwrap_deg(degrees):
    rads = np.radians(degrees)
    unwrapped = np.unwrap(rads)
    return np.degrees(unwrapped)

# --- 3. 核心逻辑 ---
def run_simulation():
    filename = os.path.basename(TARGET_FILE)
    print(f"正在读取文件: {filename} ...")
    
    df = strict_tab_parse(TARGET_FILE)
    if df is None: 
        print("读取失败，请检查文件路径。")
        return
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    # --- 提取高度层 ---
    clean_ws_cols_temp = [c for c in df.columns if ('水平风速' in c or 'Speed' in c) and is_clean_col(c)]
    raw_heights = []
    for c in clean_ws_cols_temp:
        m = re.search(r'(\d+)m', c)
        if m: raw_heights.append(int(m.group(1)))
    heights = sorted(list(set(raw_heights)))
    print(f"识别高度层: {heights}")

    # --- 准备数据矩阵 ---
    n_samples = len(df)
    n_heights = len(heights)
    
    # 存储列名的映射以便提取
    cols_ws = [get_pure_col(df.columns, h, ['水平风速', 'Speed']) for h in heights]
    cols_wd = [get_pure_col(df.columns, h, ['风向', 'Direction']) for h in heights]
    cols_std = []
    for h in heights:
        c = next((x for x in df.columns if f'{h}m' in x and ('偏差' in x or 'Std' in x) and '风向' not in x), None)
        cols_std.append(c)

    # 提取矩阵
    mat_ws = np.full((n_samples, n_heights), np.nan)
    mat_wd = np.full((n_samples, n_heights), np.nan)
    mat_ti = np.full((n_samples, n_heights), np.nan)

    for i in range(n_heights):
        if cols_ws[i]: mat_ws[:, i] = pd.to_numeric(df[cols_ws[i]], errors='coerce').values
        if cols_wd[i]: mat_wd[:, i] = pd.to_numeric(df[cols_wd[i]], errors='coerce').values
        
        # 计算 TI
        if cols_std[i] and cols_ws[i]:
            std_val = pd.to_numeric(df[cols_std[i]], errors='coerce').values
            ws_val = mat_ws[:, i]
            with np.errstate(divide='ignore', invalid='ignore'):
                ti_val = std_val / ws_val
                ti_val[ws_val < 3.0] = np.nan
            mat_ti[:, i] = ti_val

    # --- 寻找符合条件的 LLJ 个例 ---
    print("正在筛选 LLJ 个例...")
    valid_indices = []
    
    # 矢量化查找太快，这里为了逻辑清晰用循环（仅筛选）
    for i in range(n_samples):
        u = mat_ws[i, :]
        if np.isnan(u).any(): continue # 简单起见，要求数据完整
        
        idx_max = np.argmax(u)
        u_jet = u[idx_max]
        z_jet = heights[idx_max]
        
        # 判定
        if not (MIN_JET_HEIGHT <= z_jet <= MAX_JET_HEIGHT): continue
        if idx_max == 0 or idx_max == n_heights - 1: continue # 边界极值不算
        
        u_bottom = u[0]
        u_top = u[-1]
        
        if (u_jet - u_bottom >= LLJ_THRESHOLD) and (u_jet - u_top >= LLJ_THRESHOLD):
            valid_indices.append(i)
            
    print(f"找到 {len(valid_indices)} 个完整数据的急流时刻。")
    if not valid_indices: return

    # --- 随机挑选一个幸运儿 (或者您可以指定 index) ---
    chosen_idx = random.choice(valid_indices)
    # chosen_idx = 1234 # 如果你想固定看某一行
    
    timestamp = df.iloc[chosen_idx]['Date/Time']
    print(f"\n>>> 选中个例: Index={chosen_idx}, Time={timestamp} <<<")

    # --- 获取该时刻的真实数据 (Observed) ---
    obs_z = np.array(heights)
    obs_ws = mat_ws[chosen_idx, :]
    obs_wd = mat_wd[chosen_idx, :]
    obs_ti = mat_ti[chosen_idx, :]

    # 获取个例的特征参数
    idx_max = np.argmax(obs_ws)
    REAL_Z_JET = obs_z[idx_max]
    REAL_U_JET = obs_ws[idx_max]
    REAL_WD_JET = obs_wd[idx_max]

    print(f"实测特征: Z_jet={REAL_Z_JET}m, U_jet={REAL_U_JET:.2f}m/s, WD_jet={REAL_WD_JET:.1f}°")

    # --- 生成模拟数据 (Simulated) ---
    # 使用通用参数 + 个例特征
    sim_z = np.linspace(0, 500, 100)
    
    # 1. WS Sim
    sim_ws = model_ws_banta(sim_z, REAL_U_JET, REAL_Z_JET, 
                            STATION_PARAMS['WS_Alpha'], STATION_PARAMS['WS_Beta'])
    
    # 2. TI Sim
    sim_ti = model_ti_banta_inv(sim_z, REAL_Z_JET, 
                                STATION_PARAMS['TI_Base'], STATION_PARAMS['TI_Dip'],
                                STATION_PARAMS['TI_Alpha'], STATION_PARAMS['TI_Beta'])
    
    # 3. WD Sim
    # 注意：真实 WD 可能跨越 360/0，画图时不好看，需要先解缠绕
    # 这里我们只模拟"趋势"，为了画图方便，我们把观测的风向也平移/解缠绕一下
    obs_wd_unwrapped = unwrap_deg(obs_wd)
    # 重新找一下解缠绕后的 jet 处风向
    wd_jet_unwrapped = obs_wd_unwrapped[idx_max]
    
    sim_wd = model_wd_linear(sim_z, wd_jet_unwrapped, REAL_Z_JET, STATION_PARAMS['WD_Slope'])

    # --- 绘图验证 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Subplot 1: Wind Speed
    axes[0].plot(obs_ws, obs_z, 'ko', markersize=8, label='实测 (Observed)')
    axes[0].plot(sim_ws, sim_z, 'r-', linewidth=3, alpha=0.8, label='模拟 (Model)')
    axes[0].set_xlabel('Wind Speed [m/s]', fontsize=12)
    axes[0].set_ylabel('Height [m]', fontsize=12)
    axes[0].set_title(f'风速廓线 (Banta)\nInput: U_jet={REAL_U_JET:.1f}, Z_jet={REAL_Z_JET}', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    axes[0].axhline(REAL_Z_JET, color='gray', linestyle=':', alpha=0.5)

    # Subplot 2: Turbulence Intensity
    axes[1].plot(obs_ti, obs_z, 'ko', markersize=8, label='实测 (Observed)')
    axes[1].plot(sim_ti, sim_z, 'b-', linewidth=3, alpha=0.8, label='模拟 (Model)')
    axes[1].set_xlabel('TI [-]', fontsize=12)
    axes[1].set_title(f'湍流廓线 (Inv-Banta)\nParams: Base={STATION_PARAMS["TI_Base"]}, Dip={STATION_PARAMS["TI_Dip"]}', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_xlim(0, 0.4) # TI 一般不超 0.4
    axes[1].axhline(REAL_Z_JET, color='gray', linestyle=':', alpha=0.5)

    # Subplot 3: Wind Direction
    axes[2].plot(obs_wd_unwrapped, obs_z, 'ko', markersize=8, label='实测 (Observed)')
    axes[2].plot(sim_wd, sim_z, 'g-', linewidth=3, alpha=0.8, label='模拟 (Model)')
    axes[2].set_xlabel('Wind Direction [deg, unwrapped]', fontsize=12)
    axes[2].set_title(f'风向廓线 (Linear)\nSlope k={STATION_PARAMS["WD_Slope"]}', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].axhline(REAL_Z_JET, color='gray', linestyle=':', alpha=0.5)

    # Global Title
    fig.suptitle(f"低空急流个例验证 (Single Event Validation)\nStation: 双鸭山集贤 | Time: {timestamp}", fontsize=16, y=1.02)
    
    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, f'Validation_{timestamp.replace(":","-").replace(" ","_")}.png')
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"\n[完成] 验证图已保存: {out_img}")

if __name__ == "__main__":
    run_simulation()