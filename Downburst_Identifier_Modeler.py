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

# ================= 用户配置区域 =================
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/downburst_final_model'

# 1. 场站匿名映射 (文件名关键字 -> 代号)
STATION_MAPPING = {
    '大庆': 'Station 1',
    '双鸭山': 'Station 2',
    '昌图': 'Station 3',
    '台安': 'Station 4',
    '衡水': 'Station 5',
    '潍坊': 'Station 6',
    '开封': 'Station 7',
    '菏泽': 'Station 8',
    '盐城': 'Station 9',
    '孝感': 'Station 10'
}

# 2. 识别判据
MIN_GUST_SPEED = 12.0      # 最小阵风速度
MIN_GUST_FACTOR = 1.25     # 最小阵风系数 (GF = Max/Mean)
MAX_ALLOWED_DELTA = 800.0  # 最大允许射流厚度 (超过视为LLJ剔除)

# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 字体配置 ---
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

# --- 模型公式 ---
# 1. 风速: Wood-Kwok (1998)
def model_ws_wk(z, u_max, delta):
    z = np.maximum(z, 0.1)
    # 典型系数 1.55 和 0.7
    term1 = 1.55 * u_max
    term2 = np.power(z / delta, 1.0/6.0)
    term3 = 1.0 - erf(0.7 * z / delta)
    return term1 * term2 * term3

# 2. 风向: 线性模型 (Linear)
def model_wd_linear(z, offset, slope):
    return offset + slope * z

# 3. 阵风系数: 幂律模型 (Power Law)
def model_gf_power(z, a, gamma, b):
    # GF = A * z^(-gamma) + B
    return a * np.power(z, -gamma) + b

# --- 工具函数 ---
def get_station_code(filename):
    base = os.path.basename(filename)
    for key, code in STATION_MAPPING.items():
        if key in base: return code
    return base.split('-')[0]

def unwrap_deg(degrees):
    rads = np.radians(degrees)
    unwrapped = np.unwrap(rads)
    return np.degrees(unwrapped)

def is_clean_col(col_name):
    blacklist = ['最小', '偏差', '矢量', '标准差', 'Min', 'Std', 'Sigma']
    for bad in blacklist:
        if bad.lower() in col_name.lower(): return False
    return True

def get_col(cols, h, keywords):
    for c in cols:
        if f'{h}m' not in c: continue
        if not any(k in c for k in keywords): continue
        if is_clean_col(c): return c
    return None

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

# --- 核心处理逻辑 ---
def process_station(file_path):
    station_code = get_station_code(file_path)
    print(f"[{station_code}] 正在识别与建模...", end=' ')

    df = strict_tab_parse(file_path)
    if df is None: return None
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    # 1. 识别高度层
    # 优先找 Max/Gust 列
    cols_g = [c for c in df.columns if ('最大' in c or 'Gust' in c or 'Max' in c) and is_clean_col(c) and 'm' in c]
    if not cols_g: # 如果没有最大风速列，降级使用平均风速(效果会差)
        cols_g = [c for c in df.columns if ('水平' in c or 'Speed' in c) and is_clean_col(c)]
    
    heights = sorted(list(set([int(re.search(r'(\d+)m', c).group(1)) for c in cols_g if re.search(r'(\d+)m', c)])))
    if len(heights) < 4: return None

    # 2. 构建矩阵
    n, m = len(df), len(heights)
    mat_gust = np.full((n, m), np.nan)
    mat_mean = np.full((n, m), np.nan)
    mat_wd   = np.full((n, m), np.nan)

    for i, h in enumerate(heights):
        c_g = get_col(df.columns, h, ['最大', 'Gust', 'Max'])
        c_m = get_col(df.columns, h, ['水平', 'Speed'])
        c_d = get_col(df.columns, h, ['风向', 'Direction'])
        
        # Fallback
        if not c_g and c_m: c_g = c_m

        if c_g: mat_gust[:, i] = pd.to_numeric(df[c_g], errors='coerce').values
        if c_m: mat_mean[:, i] = pd.to_numeric(df[c_m], errors='coerce').values
        if c_d: mat_wd[:, i]   = pd.to_numeric(df[c_d], errors='coerce').values

    # 3. 筛选下击暴流 (Identification)
    valid_indices = []
    for i in range(n):
        row_g = mat_gust[i, :]
        row_m = mat_mean[i, :]
        if np.isnan(row_g).all(): continue
        
        # A. 强度 & 阵风系数判据
        u_max = np.nanmax(row_g)
        idx_max = np.nanargmax(row_g)
        
        if u_max < MIN_GUST_SPEED: continue
        
        # 计算 GF (Gust Factor)
        u_mean_at_max = row_m[idx_max]
        gf = u_max / u_mean_at_max if u_mean_at_max > 1 else 1.0
        if gf < MIN_GUST_FACTOR: continue
        
        # B. 形状判据 (要有鼻子)
        if idx_max == 0 or idx_max == m-1: continue
        if not (row_g[idx_max] > row_g[0]+0.5 and row_g[idx_max] > row_g[-1]+0.5): continue

        valid_indices.append(i)

    # 4. 收集数据 & 终极筛选 (射流厚度)
    collected_ws = []
    collected_wd_twist = []
    collected_gf = []

    for idx in valid_indices:
        y_g = mat_gust[idx, :]
        mask = ~np.isnan(y_g)
        if np.sum(mask) < 4: continue
        
        # C. 终极判据: Wood-Kwok 拟合厚度 (剔除 LLJ)
        try:
            popt, _ = curve_fit(model_ws_wk, heights, y_g, p0=[np.max(y_g), 300], bounds=([5,10], [100,2500]))
            if popt[1] > MAX_ALLOWED_DELTA: continue # 剔除
        except: continue
        
        # 通过所有测试，加入集合
        collected_ws.append(y_g)
        
        # 收集风向扭曲
        raw_wd = mat_wd[idx, :]
        if not np.isnan(raw_wd[0]):
            unwrapped = unwrap_deg(raw_wd)
            collected_wd_twist.append(unwrapped - unwrapped[0])
            
        # 收集 GF
        with np.errstate(divide='ignore', invalid='ignore'):
            gf_prof = y_g / mat_mean[idx, :]
            gf_prof[mat_mean[idx,:] < 2.0] = np.nan
            gf_prof[gf_prof > 3.0] = np.nan # 剔除异常点
        collected_gf.append(gf_prof)

    count = len(collected_ws)
    print(f"最终确认样本: {count}")
    if count < 2: return None

    # 5. 建模拟合 (Modeling)
    avg_ws = np.nanmean(np.array(collected_ws), axis=0)
    avg_wd = np.nanmean(np.array(collected_wd_twist), axis=0)
    avg_gf = np.nanmean(np.array(collected_gf), axis=0)

    # Fit WS (Wood-Kwok)
    try: p_ws, _ = curve_fit(model_ws_wk, heights, avg_ws, p0=[15, 400]); ws_r2 = 1.0 # 简化R2
    except: p_ws = [np.nan, np.nan]

    # Fit WD (Linear)
    try: 
        mask_wd = ~np.isnan(avg_wd)
        p_wd, _ = curve_fit(model_wd_linear, np.array(heights)[mask_wd], avg_wd[mask_wd], p0=[0, 0])
    except: p_wd = [0, 0]

    # Fit GF (Power Law)
    try:
        mask_gf = ~np.isnan(avg_gf)
        p_gf, _ = curve_fit(model_gf_power, np.array(heights)[mask_gf], avg_gf[mask_gf], p0=[1.0, 0.1, 1.2])
    except: p_gf = [np.nan, np.nan, np.nan]

    # 6. 绘图
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    z_smooth = np.linspace(0, heights[-1], 100)
    
    # WS
    axes[0].plot(avg_ws, heights, 'ko', label='Obs Avg')
    if not np.isnan(p_ws[0]):
        axes[0].plot(model_ws_wk(z_smooth, *p_ws), z_smooth, 'r-', lw=2, label='Wood-Kwok Model')
        axes[0].set_title(f"Gust Profile\nUmax={p_ws[0]:.1f}, Delta={p_ws[1]:.0f}m")
    axes[0].set_xlabel('Max Wind Speed [m/s]')
    axes[0].set_ylabel('Height [m]')
    axes[0].legend()
    axes[0].grid(True)

    # WD
    axes[1].plot(avg_wd, heights, 'ko')
    axes[1].plot(model_wd_linear(z_smooth, *p_wd), z_smooth, 'g-', lw=2, label='Linear Model')
    axes[1].set_title(f"WD Twist (Relative)\nSlope={p_wd[1]:.3f} deg/m")
    axes[1].set_xlabel('Twist [deg]')
    axes[1].grid(True)

    # GF
    axes[2].plot(avg_gf, heights, 'ko')
    if not np.isnan(p_gf[0]):
        axes[2].plot(model_gf_power(z_smooth, *p_gf), z_smooth, 'b-', lw=2, label='Power Law Model')
        axes[2].set_title(f"Gust Factor\nGF={p_gf[0]:.2f}*z^(-{p_gf[1]:.2f})+{p_gf[2]:.2f}")
    axes[2].set_xlabel('Gust Factor [-]')
    axes[2].grid(True)

    plt.suptitle(f"{station_code} Downburst Identification & Model (N={count})", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{station_code}_Model.png'))
    plt.close()

    return {
        'Station': station_code,
        'Count': count,
        'WK_Umax': p_ws[0], 'WK_Delta': p_ws[1],
        'WD_Slope': p_wd[1],
        'GF_A': p_gf[0], 'GF_Gamma': p_gf[1], 'GF_Base': p_gf[2]
    }

# --- 主程序 ---
def main():
    print("="*60)
    print(" 下击暴流识别与风模型构建 (Downburst Identification & Modeling)")
    print("="*60)

    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    
    # 排序：按映射后的 Station 1-10 顺序处理
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
        out_excel = os.path.join(OUTPUT_DIR, 'Downburst_Model_Parameters.xlsx')
        df.to_excel(out_excel, index=False)
        print(f"\n全部完成！结果已保存至: {out_excel}")
    else:
        print("\n未提取到有效模型参数。")

if __name__ == "__main__":
    main()