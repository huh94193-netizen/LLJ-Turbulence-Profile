import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy.special import erf
import re
import os
import glob
import warnings

# ================= 配置区域 =================
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/downburst_robust_gf'

# 判据保持不变
MIN_GUST_SPEED = 12.0      
MIN_GUST_FACTOR = 1.25     
MAX_ALLOWED_DELTA = 800.0
TARGET_MONTHS = [5, 6, 7, 8, 9]   
TARGET_HOURS = list(range(9, 24)) 
TEMP_DROP_THRESHOLD = -1.0 
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 字体 ---
def get_safe_font_prop():
    candidates = ['WenQuanYi Micro Hei', 'Zen Hei', 'SimHei', 'Microsoft YaHei']
    for font in candidates:
        try:
            if fm.findfont(font) != fm.findfont('DejaVu Sans'): return font
        except: continue
    return None
SYSTEM_ZH_FONT = get_safe_font_prop()
if SYSTEM_ZH_FONT:
    plt.rcParams['font.sans-serif'] = [SYSTEM_ZH_FONT]
    plt.rcParams['axes.unicode_minus'] = False

# --- 模型公式 ---
# 1. 风速: Wood-Kwok
def model_ws_wk(z, u_max, delta):
    z = np.maximum(z, 0.1)
    return 1.55 * u_max * np.power(z/delta, 1/6) * (1 - erf(0.7*z/delta))

# 2. 阵风系数 (受限幂律): 强制底数为 1.15
def model_gf_power_constrained(z, a, gamma):
    # 固定 B = 1.15 (根据经验，高空 GF 不会低于 1.1-1.2)
    return a * np.power(z, -gamma) + 1.15

# 3. 阵风系数 (线性兜底)
def model_gf_linear(z, slope, intercept):
    return slope * z + intercept

# --- 工具函数 ---
def get_station_code(filename):
    # 简单的映射逻辑
    MAPPING = {
        '大庆': 'Station 1', '双鸭山': 'Station 2', '昌图': 'Station 3',
        '台安': 'Station 4', '衡水': 'Station 5', '潍坊': 'Station 6',
        '开封': 'Station 7', '菏泽': 'Station 8', '盐城': 'Station 9',
        '孝感': 'Station 10'
    }
    base = os.path.basename(filename)
    for key, code in MAPPING.items():
        if key in base: return code
    return base.split('-')[0]

def is_clean_col(col_name):
    blacklist = ['最小', '偏差', '矢量', '标准差', 'Min', 'Std', 'Sigma']
    for bad in blacklist:
        if bad.lower() in col_name.lower(): return False
    return True

def strict_tab_parse(file_path):
    encodings = ['utf-8', 'gbk', 'gb18030', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, sep='\t', skiprows=12, encoding=enc, engine='python')
            if 'Date' in str(df.columns) or '时间' in str(df.columns): return df
            df = pd.read_csv(file_path, sep='\s+', skiprows=12, encoding=enc, engine='python')
            if 'Date' in str(df.columns): return df
        except: continue
    return None

# --- 处理逻辑 ---
def process_station(file_path):
    station_code = get_station_code(file_path)
    print(f"[{station_code}] 处理中...", end=' ')

    df = strict_tab_parse(file_path)
    if df is None: return None
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    # 解析时间
    try: df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    except: return None

    # 获取列
    cols_g = [c for c in df.columns if ('最大' in c or 'Gust' in c or 'Max' in c) and is_clean_col(c) and 'm' in c]
    if not cols_g: cols_g = [c for c in df.columns if ('水平' in c or 'Speed' in c) and is_clean_col(c)]
    
    # 提取高度
    heights = sorted(list(set([int(re.search(r'(\d+)m', c).group(1)) for c in cols_g if re.search(r'(\d+)m', c)])))
    if len(heights) < 4: return None

    # 气温列
    col_temp = next((c for c in df.columns if '外温' in c or 'Temperature' in c), None)
    temp_diff = np.zeros(len(df))
    if col_temp:
        temp_series = pd.to_numeric(df[col_temp], errors='coerce')
        temp_diff = temp_series.diff(2).fillna(0).values

    # 构建矩阵
    n, m = len(df), len(heights)
    mat_gust = np.full((n, m), np.nan)
    mat_mean = np.full((n, m), np.nan)

    for i, h in enumerate(heights):
        # 简化版找列逻辑
        c_g = next((c for c in df.columns if f'{h}m' in c and ('最大' in c or 'Gust' in c) and is_clean_col(c)), None)
        c_m = next((c for c in df.columns if f'{h}m' in c and ('水平' in c or 'Speed' in c) and is_clean_col(c)), None)
        if not c_g and c_m: c_g = c_m
        
        if c_g: mat_gust[:, i] = pd.to_numeric(df[c_g], errors='coerce').values
        if c_m: mat_mean[:, i] = pd.to_numeric(df[c_m], errors='coerce').values

    # 筛选
    valid_indices = []
    months = df['Date/Time'].dt.month.values
    hours = df['Date/Time'].dt.hour.values

    for i in range(n):
        if months[i] not in TARGET_MONTHS: continue
        if hours[i] not in TARGET_HOURS: continue
        if col_temp and (temp_diff[i] > TEMP_DROP_THRESHOLD): continue

        row_g = mat_gust[i, :]
        row_m = mat_mean[i, :]
        if np.isnan(row_g).all(): continue
        
        u_max = np.nanmax(row_g)
        if u_max < MIN_GUST_SPEED: continue
        
        idx_max = np.nanargmax(row_g)
        u_mean_val = row_m[idx_max]
        gf = u_max / u_mean_val if u_mean_val > 1 else 1.0
        if gf < MIN_GUST_FACTOR: continue
        
        if idx_max == 0 or idx_max == m-1: continue
        if not (row_g[idx_max] > row_g[0]+0.5 and row_g[idx_max] > row_g[-1]+0.5): continue
        
        valid_indices.append(i)

    # Wood-Kwok 剔除 LLJ
    collected_ws = []
    collected_gf = []

    for idx in valid_indices:
        y_g = mat_gust[idx, :]
        mask = ~np.isnan(y_g)
        if np.sum(mask) < 4: continue
        
        try:
            popt, _ = curve_fit(model_ws_wk, heights, y_g, p0=[np.max(y_g), 300], bounds=([5,10], [100,2500]))
            if popt[1] > MAX_ALLOWED_DELTA: continue 
        except: continue
        
        collected_ws.append(y_g)
        
        # 计算 GF
        with np.errstate(divide='ignore', invalid='ignore'):
            gf_prof = y_g / mat_mean[idx, :]
            gf_prof[mat_mean[idx,:] < 2.0] = np.nan
            gf_prof[gf_prof > 4.0] = np.nan # 剔除离谱值
        collected_gf.append(gf_prof)

    count = len(collected_ws)
    print(f"最终样本: {count}", end=' ')
    if count < 2: 
        print("(样本不足)")
        return None

    # 平均
    avg_ws = np.nanmean(np.array(collected_ws), axis=0)
    avg_gf = np.nanmean(np.array(collected_gf), axis=0)

    # === 1. 风速拟合 (Wood-Kwok) ===
    try: p_ws, _ = curve_fit(model_ws_wk, heights, avg_ws, p0=[15, 400])
    except: p_ws = [np.nan, np.nan]

    # === 2. 阵风系数拟合 (智能降级策略) ===
    gf_model_type = "Power Law (Constrained)"
    p_gf = [np.nan, np.nan] # [A, Gamma] or [Slope, Intercept]
    
    # 尝试策略 A: 受限幂律 (Fixed B=1.15)
    mask_gf = ~np.isnan(avg_gf)
    x_valid = np.array(heights)[mask_gf]
    y_valid = avg_gf[mask_gf]
    
    try:
        # p0=[A, gamma], bounds: A>0, gamma>0
        popt_pow, _ = curve_fit(model_gf_power_constrained, x_valid, y_valid, 
                                p0=[1.0, 0.1], bounds=([0, 0], [10, 2.0]))
        p_gf = popt_pow
        print("| GF拟合: 幂律成功")
    except:
        # 策略 A 失败，转策略 B: 线性
        try:
            popt_lin, _ = curve_fit(model_gf_linear, x_valid, y_valid, p0=[-0.001, 1.5])
            p_gf = popt_lin
            gf_model_type = "Linear (Fallback)"
            print("| GF拟合: 降级为线性")
        except:
            print("| GF拟合: 失败")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    z_smooth = np.linspace(0, heights[-1], 100)
    
    # Left: Wind Speed
    axes[0].plot(avg_ws, heights, 'ko', label='Obs')
    if not np.isnan(p_ws[0]):
        axes[0].plot(model_ws_wk(z_smooth, *p_ws), z_smooth, 'r-', lw=2, label='Wood-Kwok')
        axes[0].set_title(f"Max Gust Profile\n$U_{{max}}$={p_ws[0]:.1f}, $\delta$={p_ws[1]:.0f}")
    axes[0].set_ylabel('Height [m]')
    axes[0].set_xlabel('m/s')
    axes[0].grid(True)
    axes[0].legend()

    # Right: Gust Factor
    axes[1].plot(avg_gf, heights, 'ko', label='Obs')
    if not np.isnan(p_gf[0]):
        if "Power" in gf_model_type:
            y_pred = model_gf_power_constrained(z_smooth, *p_gf)
            label_str = f"Power: $GF={p_gf[0]:.2f}z^{{-{p_gf[1]:.2f}}}+1.15$"
        else:
            y_pred = model_gf_linear(z_smooth, *p_gf)
            label_str = f"Linear: Slope={p_gf[0]:.4f}"
            
        axes[1].plot(y_pred, z_smooth, 'b-', lw=2, label=label_str)
        axes[1].set_title(f"Gust Factor ({gf_model_type})")
        
    axes[1].set_xlabel('GF [-]')
    axes[1].grid(True)
    axes[1].legend()

    plt.suptitle(f"{station_code} Robust Downburst Model (N={count})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{station_code}_RobustModel.png'))
    plt.close()

    return {
        'Station': station_code,
        'Count': count,
        'WK_Umax': p_ws[0], 'WK_Delta': p_ws[1],
        'GF_Model': gf_model_type,
        'GF_Param1': p_gf[0], 'GF_Param2': p_gf[1]
    }

# --- 主程序 ---
def main():
    print("="*60)
    print(" 下击暴流强壮版建模 (GF 智能降级)")
    print("="*60)
    
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    
    def get_sort_key(fname):
        code = get_station_code(fname)
        m = re.search(r'(\d+)', code)
        return int(m.group(1)) if m else 999
    files.sort(key=get_sort_key)
    
    results = []
    for f in files:
        res = process_station(f)
        if res: results.append(res)
        
    if results:
        df = pd.DataFrame(results)
        df.to_excel(os.path.join(OUTPUT_DIR, 'Downburst_Robust_Params.xlsx'), index=False)
        print("\n完成！")

if __name__ == "__main__":
    main()