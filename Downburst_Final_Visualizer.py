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

# ================= 最终配置区域 =================
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/downburst_final_visualization'

# 1. 场站映射 (匿名化)
STATION_MAPPING = {
    '大庆': 'Station 1', '双鸭山': 'Station 2', '昌图': 'Station 3',
    '台安': 'Station 4', '衡水': 'Station 5', '潍坊': 'Station 6',
    '开封': 'Station 7', '菏泽': 'Station 8', '盐城': 'Station 9',
    '孝感': 'Station 10'
}

# 2. 严格筛选标准 (保真不保量)
MIN_GUST_SPEED = 12.0      
MIN_GUST_FACTOR = 1.25     
MAX_ALLOWED_DELTA = 800.0  # 核心判据
TARGET_MONTHS = [5, 6, 7, 8, 9]   
TARGET_HOURS = list(range(9, 24)) 
TEMP_DROP_THRESHOLD = -1.0 # 气温骤降判据
# ===============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 字体设置 ---
def get_safe_font_prop():
    candidates = ['WenQuanYi Micro Hei', 'Zen Hei', 'SimHei', 'Microsoft YaHei', 'SimSun']
    for font in candidates:
        try:
            if fm.findfont(font) != fm.findfont('DejaVu Sans'): return font
        except: continue
    return None
SYSTEM_ZH_FONT = get_safe_font_prop()
if SYSTEM_ZH_FONT:
    plt.rcParams['font.sans-serif'] = [SYSTEM_ZH_FONT]
    plt.rcParams['axes.unicode_minus'] = False

# --- Wood-Kwok 模型 ---
def model_ws_wk(z, u_max, delta):
    z = np.maximum(z, 0.1)
    # 典型系数 1.55 和 0.7
    return 1.55 * u_max * np.power(z/delta, 1/6) * (1 - erf(0.7*z/delta))

# --- 工具函数 ---
def get_station_code(filename):
    base = os.path.basename(filename)
    for key, code in STATION_MAPPING.items():
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

# --- 核心处理 ---
def process_and_plot(file_path):
    station_code = get_station_code(file_path)
    print(f"[{station_code}]Processing...", end=' ')

    df = strict_tab_parse(file_path)
    if df is None: return None
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    try: df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    except: return None

    # 1. 识别高度 & 气温
    cols_g = [c for c in df.columns if ('最大' in c or 'Gust' in c or 'Max' in c) and is_clean_col(c) and 'm' in c]
    if not cols_g: cols_g = [c for c in df.columns if ('水平' in c or 'Speed' in c) and is_clean_col(c)]
    heights = sorted(list(set([int(re.search(r'(\d+)m', c).group(1)) for c in cols_g if re.search(r'(\d+)m', c)])))
    if len(heights) < 4: return None

    col_temp = next((c for c in df.columns if '外温' in c or 'Temperature' in c), None)
    temp_diff = np.zeros(len(df))
    if col_temp:
        temp_s = pd.to_numeric(df[col_temp], errors='coerce')
        temp_diff = temp_s.diff(2).fillna(0).values

    # 2. 构建矩阵
    n, m = len(df), len(heights)
    mat_gust = np.full((n, m), np.nan)
    mat_mean = np.full((n, m), np.nan) # 仅用于GF筛选

    for i, h in enumerate(heights):
        c_g = next((c for c in df.columns if f'{h}m' in c and ('最大' in c or 'Gust' in c) and is_clean_col(c)), None)
        c_m = next((c for c in df.columns if f'{h}m' in c and ('水平' in c or 'Speed' in c) and is_clean_col(c)), None)
        if not c_g and c_m: c_g = c_m
        
        if c_g: mat_gust[:, i] = pd.to_numeric(df[c_g], errors='coerce').values
        if c_m: mat_mean[:, i] = pd.to_numeric(df[c_m], errors='coerce').values

    # 3. 筛选循环
    valid_profiles = [] # 存储通过筛选的真实廓线
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
        
        # 形状: 鼻尖不能在两头
        if idx_max == 0 or idx_max == m-1: continue
        if not (row_g[idx_max] > row_g[0]+0.5 and row_g[idx_max] > row_g[-1]+0.5): continue

        # Wood-Kwok 单条拟合筛选 (Delta)
        try:
            mask = ~np.isnan(row_g)
            if np.sum(mask) < 4: continue
            popt, _ = curve_fit(model_ws_wk, np.array(heights)[mask], row_g[mask], 
                                p0=[u_max, 300], bounds=([5,10], [100,2500]))
            if popt[1] > MAX_ALLOWED_DELTA: continue
            
            # 收集该条廓线
            valid_profiles.append(row_g)
        except: continue

    count = len(valid_profiles)
    print(f"N={count}")
    if count < 1: return None

    # 4. 整体平均与拟合
    valid_profiles = np.array(valid_profiles)
    avg_profile = np.nanmean(valid_profiles, axis=0)

    try:
        popt, _ = curve_fit(model_ws_wk, heights, avg_profile, p0=[15, 300])
        wk_u_max = popt[0]
        wk_delta = popt[1]
        wk_r2 = 1 - np.sum((avg_profile - model_ws_wk(heights, *popt))**2) / np.sum((avg_profile - np.mean(avg_profile))**2)
    except:
        wk_u_max, wk_delta, wk_r2 = np.nan, np.nan, np.nan

    # 5. 作图 (Spaghetti Plot)
    plt.figure(figsize=(7, 9))
    
    # A. 画所有筛选出来的真实廓线 (灰色细线)
    z_smooth = np.linspace(0, heights[-1]*1.1, 100)
    
    for prof in valid_profiles:
        plt.plot(prof, heights, color='gray', alpha=0.3, linewidth=1)
    
    # B. 画平均观测值 (黑色散点)
    plt.plot(avg_profile, heights, 'ko', label='Observed Mean', zorder=5, markersize=8)
    
    # C. 画 Wood-Kwok 拟合曲线 (红色粗线)
    if not np.isnan(wk_u_max):
        plt.plot(model_ws_wk(z_smooth, *popt), z_smooth, 'r-', linewidth=3, label='Wood-Kwok Model', zorder=10)
        
        # 标注鼻尖高度
        nose_height = wk_delta * 0.23 # Wood-Kwok模型中 Zmax approx 0.23*delta
        plt.axhline(nose_height, color='r', linestyle='--', alpha=0.5)
        plt.text(np.min(avg_profile)*0.9, nose_height+5, f'Nose ~ {nose_height:.0f}m', color='r', fontsize=9)

    # 装饰
    plt.title(f"{station_code} Downburst Gust Profile\n(N={count} Events)", fontsize=14, fontweight='bold')
    plt.xlabel('Gust Wind Speed [m/s]', fontsize=12)
    plt.ylabel('Height [m]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 统计信息框
    info_text = (
        f"Model Parameters:\n"
        f"  $U_{{max}}$ = {wk_u_max:.1f} m/s\n"
        f"  $\delta$ = {wk_delta:.0f} m\n"
        f"  $R^2$ = {wk_r2:.2f}"
    )
    plt.gca().text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{station_code}_Final_Profile.png'), dpi=300)
    plt.close()

    return {
        'Station': station_code,
        'Count': count,
        'WK_Umax': wk_u_max,
        'WK_Delta': wk_delta,
        'R2': wk_r2
    }

# --- 主程序 ---
def main():
    print("="*60)
    print(" 下击暴流最终建模 (真实案例集 + Wood-Kwok拟合)")
    print("="*60)
    
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    
    # 排序
    def get_sort_key(fname):
        code = get_station_code(fname)
        m = re.search(r'(\d+)', code)
        return int(m.group(1)) if m else 999
    files.sort(key=get_sort_key)
    
    results = []
    for f in files:
        res = process_and_plot(f)
        if res: results.append(res)
        
    if results:
        df = pd.DataFrame(results)
        df.to_excel(os.path.join(OUTPUT_DIR, 'Downburst_Final_Params.xlsx'), index=False)
        print(f"\n全部完成！结果已保存在 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()