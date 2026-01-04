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
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/downburst_custom_model'

# 场站映射
STATION_MAPPING = {
    '大庆': 'Station 1', '双鸭山': 'Station 2', '昌图': 'Station 3',
    '台安': 'Station 4', '衡水': 'Station 5', '潍坊': 'Station 6',
    '开封': 'Station 7', '菏泽': 'Station 8', '盐城': 'Station 9',
    '孝感': 'Station 10'
}

# 筛选标准 (保持之前严格的那一套)
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

# --- 【核心修改】广义 Wood-Kwok 模型 ---
# 将 1/6 替换为变量 alpha
def model_wk_custom(z, u_max, delta, alpha):
    z = np.maximum(z, 0.1)
    # 保持 1.55 和 0.7 不变，只让 alpha 动，这样物理意义最清晰
    # alpha 越大，廓线“胖”得越快，对应地面越粗糙
    return 1.55 * u_max * np.power(z/delta, alpha) * (1 - erf(0.7*z/delta))

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

# --- 处理逻辑 ---
def process_station(file_path):
    station_code = get_station_code(file_path)
    print(f"[{station_code}] 定制化拟合中...", end=' ')

    df = strict_tab_parse(file_path)
    if df is None: return None
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    
    try: df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    except: return None

    # 获取列
    cols_g = [c for c in df.columns if ('最大' in c or 'Gust' in c or 'Max' in c) and is_clean_col(c) and 'm' in c]
    if not cols_g: cols_g = [c for c in df.columns if ('水平' in c or 'Speed' in c) and is_clean_col(c)]
    heights = sorted(list(set([int(re.search(r'(\d+)m', c).group(1)) for c in cols_g if re.search(r'(\d+)m', c)])))
    if len(heights) < 4: return None

    # 气温
    col_temp = next((c for c in df.columns if '外温' in c or 'Temperature' in c), None)
    temp_diff = np.zeros(len(df))
    if col_temp:
        temp_s = pd.to_numeric(df[col_temp], errors='coerce')
        temp_diff = temp_s.diff(2).fillna(0).values

    # 矩阵
    n, m = len(df), len(heights)
    mat_gust = np.full((n, m), np.nan)
    mat_mean = np.full((n, m), np.nan)

    for i, h in enumerate(heights):
        c_g = next((c for c in df.columns if f'{h}m' in c and ('最大' in c or 'Gust' in c) and is_clean_col(c)), None)
        c_m = next((c for c in df.columns if f'{h}m' in c and ('水平' in c or 'Speed' in c) and is_clean_col(c)), None)
        if not c_g and c_m: c_g = c_m
        if c_g: mat_gust[:, i] = pd.to_numeric(df[c_g], errors='coerce').values
        if c_m: mat_mean[:, i] = pd.to_numeric(df[c_m], errors='coerce').values

    # 筛选
    collected_profiles = []
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
        if u_mean_val > 1 and (u_max/u_mean_val) < MIN_GUST_FACTOR: continue
        
        if idx_max == 0 or idx_max == m-1: continue
        
        # 粗筛 Delta (用标准模型先筛一遍假货)
        try:
            mask = ~np.isnan(row_g)
            if np.sum(mask) < 4: continue
            # 这里先用标准 1/6 筛，防止 alpha 乱飘导致假货入选
            def std_wk(z, u, d): return 1.55*u*(z/d)**(1/6)*(1-erf(0.7*z/d))
            p_check, _ = curve_fit(std_wk, np.array(heights)[mask], row_g[mask], p0=[u_max, 300])
            if p_check[1] > MAX_ALLOWED_DELTA: continue
            
            collected_profiles.append(row_g)
        except: continue

    count = len(collected_profiles)
    print(f"N={count}", end=' ')
    if count < 1: return None

    # 平均
    avg_profile = np.nanmean(np.array(collected_profiles), axis=0)

    # === 【关键步骤】3参数定制化拟合 ===
    # p0 = [Umax, Delta, Alpha]
    # bounds: Alpha 限制在 0.05 到 0.4 之间 (太小不物理，太大变成直线了)
    try:
        popt, pcov = curve_fit(model_wk_custom, heights, avg_profile, 
                               p0=[15, 300, 0.16], 
                               bounds=([5, 10, 0.05], [100, 2500, 0.45]))
        
        final_u = popt[0]
        final_delta = popt[1]
        final_alpha = popt[2] # 这就是你要卖给客户的“创新参数”
        
        # 计算 R2
        residuals = avg_profile - model_wk_custom(heights, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((avg_profile - np.mean(avg_profile))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"| Alpha={final_alpha:.3f} (R2={r2:.2f})")
    except:
        print("拟合失败")
        return None

    # 绘图
    plt.figure(figsize=(7, 8))
    z_smooth = np.linspace(0, heights[-1]*1.1, 100)
    
    # 画实测
    plt.plot(avg_profile, heights, 'ko', label='Observed Mean', markersize=8)
    
    # 画拟合
    y_pred = model_wk_custom(z_smooth, final_u, final_delta, final_alpha)
    plt.plot(y_pred, z_smooth, 'r-', linewidth=3, label=f'Custom Model ($\\alpha={final_alpha:.3f}$)')
    
    # 画标准对比 (可选，展示你的模型比标准的更好)
    y_std = model_wk_custom(z_smooth, final_u, final_delta, 1/6)
    plt.plot(y_std, z_smooth, 'b--', linewidth=1.5, alpha=0.6, label='Standard WK ($\\alpha=1/6$)')

    plt.title(f"{station_code} Customized Downburst Model\nShape Parameter $\\alpha = {final_alpha:.3f}$", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel('Gust Speed [m/s]')
    plt.ylabel('Height [m]')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{station_code}_CustomFit.png'))
    plt.close()

    return {
        'Station': station_code,
        'Count': count,
        'Param_Umax': final_u,
        'Param_Delta': final_delta,
        'Param_Alpha': final_alpha, # 重点参数
        'R2': r2
    }

# --- 主程序 ---
def main():
    print("="*60)
    print(" 下击暴流定制化参数拟合 (Custom Alpha Fit)")
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
        df.to_excel(os.path.join(OUTPUT_DIR, 'Downburst_Custom_Params.xlsx'), index=False)
        print(f"\n完成！定制化参数表已生成。")

if __name__ == "__main__":
    main()