import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import os
import random
import warnings

# ================= 用户配置区域 =================
# 1. 原始测风数据文件路径 (你想验证哪个场站，就填哪个文件)
#TARGET_FILE = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
TARGET_FILE = r'/home/huxun/02_LLJ/exported_data/孝感应城-1467#-20240531-20251222-filter-Exported.txt'
# 2. 【新增】参数汇总表 Excel 路径 (代码会自动去这里查参数)
PARAMS_FILE = r'/home/huxun/02_LLJ/result/all_stations_params_final/All_Stations_Parameters_Final.xlsx'

# 3. 输出路径
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/single_case_validation'

# 4. 一次性生成多少张图？
NUM_PLOTS = 10  

# 5. 判定标准 (需与之前保持一致)
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

# --- 1. 【核心新增】自动读取 Excel 参数 ---
def load_params_from_excel(excel_path, target_filename):
    """
    更健壮的参数查找逻辑：同时匹配中文名和ID
    """
    print(f"正在读取参数表: {os.path.basename(excel_path)} ...")
    if not os.path.exists(excel_path):
        print(f"[Fatal] 参数文件不存在: {excel_path}")
        return None
        
    try:
        df_params = pd.read_excel(excel_path)
        # 强制把 Station 列转为字符串，防止如果是纯数字ID被当成 int
        df_params['Station'] = df_params['Station'].astype(str)
    except Exception as e:
        print(f"[Error] 无法读取 Excel 表: {e}")
        return None

    base_name = os.path.basename(target_filename)
    
    # 1. 提取中文名 (第一段) -> "双鸭山集贤"
    name_key = base_name.split('-')[0]
    
    # 2. 提取 ID (包含数字和#) -> "1443#"
    match_id = re.search(r'(\d+#)', base_name)
    id_key = match_id.group(1) if match_id else "NON_EXISTENT_ID"
    
    print(f"尝试在 Excel 中匹配: Name='{name_key}' 或 ID='{id_key}' ...")
    
    # 3. 双重匹配查找
    # 逻辑：Station 列如果包含 name_key 或者 包含 id_key，都算对
    mask = df_params['Station'].str.contains(name_key, regex=False) | \
           df_params['Station'].str.contains(id_key, regex=False)
           
    matched_rows = df_params[mask]
    
    if matched_rows.empty:
        print(f"[Error] 未找到匹配的场站！")
        print(f"  当前 Excel 中的场站列表: {df_params['Station'].unique()}")
        return None
    
    # 取第一个匹配项
    row = matched_rows.iloc[0]
    print(f"✅ 成功匹配到场站: {row['Station']}")
    
    # 构建参数字典 (增加健壮性，防止列名缺失)
    params = {}
    cols_map = {
        'WS_Alpha': 'WS_Alpha', 'WS_Beta': 'WS_Beta',
        'TI_Base': 'TI_Base', 'TI_Dip': 'TI_Dip',
        'TI_Alpha': 'TI_Alpha', 'TI_Beta': 'TI_Beta',
        'WD_Slope': 'WD_Linear_k'
    }
    
    for key, col in cols_map.items():
        if col in row:
            params[key] = row[col]
        else:
            print(f"  [Warn] 缺少列 '{col}'，使用默认值")
            params[key] = 1.0 if 'Alpha' in key or 'Beta' in key else 0.1
            
    return params
    
# --- 2. 模型公式 ---
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

# --- 3. 数据读取工具 ---
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

# --- 4. 主逻辑 ---
def run_auto_validation():
    # 1. 自动读取参数
    station_params = load_params_from_excel(PARAMS_FILE, TARGET_FILE)
    if station_params is None:
        return # 参数读取失败则终止

    # 2. 读取原始数据
    filename = os.path.basename(TARGET_FILE)
    print(f"\n读取测风数据: {filename} ...")
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
    print("正在筛选急流个例...")
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
            
    print(f"符合条件的样本数: {len(valid_indices)}")
    if not valid_indices: return

    # --- 随机抽样 ---
    n_to_plot = min(NUM_PLOTS, len(valid_indices))
    selected_indices = random.sample(valid_indices, n_to_plot)
    
    print(f"\n>>> 开始生成 {n_to_plot} 张验证图 <<<")

    # --- 循环绘图 ---
    for count, idx in enumerate(selected_indices):
        timestamp = df.iloc[idx]['Date/Time']
        print(f"[{count+1}/{n_to_plot}] 时刻: {timestamp}")
        
        obs_z = np.array(heights)
        obs_ws = mat_ws[idx, :]
        obs_wd = mat_wd[idx, :]
        obs_ti = mat_ti[idx, :]
        
        idx_max = np.argmax(obs_ws)
        REAL_Z_JET = obs_z[idx_max]
        REAL_U_JET = obs_ws[idx_max]
        
        # 使用【自动读取的参数】生成模拟数据
        sim_z = np.linspace(0, 500, 100)
        
        # WS
        sim_ws = model_ws_banta(sim_z, REAL_U_JET, REAL_Z_JET, 
                                station_params['WS_Alpha'], station_params['WS_Beta'])
        # TI
        sim_ti = model_ti_banta_inv(sim_z, REAL_Z_JET, 
                                    station_params['TI_Base'], station_params['TI_Dip'],
                                    station_params['TI_Alpha'], station_params['TI_Beta'])
        # WD
        obs_wd_unwrapped = unwrap_deg(obs_wd)
        wd_jet_unwrapped = obs_wd_unwrapped[idx_max]
        sim_wd = model_wd_linear(sim_z, wd_jet_unwrapped, REAL_Z_JET, station_params['WD_Slope'])

        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # WS Plot
        axes[0].plot(obs_ws, obs_z, 'ko', markersize=8, label='Observed')
        axes[0].plot(sim_ws, sim_z, 'r-', lw=3, alpha=0.8, label='Model (Banta)')
        axes[0].set_title(f'WS (U={REAL_U_JET:.1f}, Z={REAL_Z_JET})')
        axes[0].set_ylabel('Height [m]')
        axes[0].grid(True, ls='--', alpha=0.5)
        axes[0].legend()
        
        # TI Plot
        axes[1].plot(obs_ti, obs_z, 'ko', markersize=8)
        axes[1].plot(sim_ti, sim_z, 'b-', lw=3, alpha=0.8, label='Model (Inv-Banta)')
        axes[1].set_title('TI Profile')
        axes[1].grid(True, ls='--', alpha=0.5)
        axes[1].set_xlim(0, 0.4)
        axes[1].legend()
        
        # WD Plot
        axes[2].plot(obs_wd_unwrapped, obs_z, 'ko', markersize=8)
        axes[2].plot(sim_wd, sim_z, 'g-', lw=3, alpha=0.8, label='Model (Linear)')
        axes[2].set_title('WD Profile')
        axes[2].grid(True, ls='--', alpha=0.5)
        axes[2].legend()
        
        safe_time = timestamp.replace(":","-").replace(" ","_").replace("/","-")
        fig.suptitle(f"Case Validation: {safe_time} | Auto-Params from Excel", fontsize=14)
        
        plt.tight_layout()
        out_name = os.path.join(OUTPUT_DIR, f'AutoVal_{count+1:02d}_{safe_time}.png')
        plt.savefig(out_name, dpi=100)
        plt.close(fig)

    print(f"\n全部完成！结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_auto_validation()