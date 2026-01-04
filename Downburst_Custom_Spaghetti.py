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
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/downburst_custom_spaghetti'

# 1. 场站映射
STATION_MAPPING = {
    '大庆': 'Station 1', '双鸭山': 'Station 2', '昌图': 'Station 3',
    '台安': 'Station 4', '衡水': 'Station 5', '潍坊': 'Station 6',
    '开封': 'Station 7', '菏泽': 'Station 8', '盐城': 'Station 9',
    '孝感': 'Station 10'
}

# 2. 严格筛选标准
MIN_GUST_SPEED = 12.0
MIN_GUST_FACTOR = 1.25
MAX_ALLOWED_DELTA = 800.0  # 筛选时用标准模型卡这一关
TARGET_MONTHS = [5, 6, 7, 8, 9]
TARGET_HOURS = list(range(9, 24))
TEMP_DROP_THRESHOLD = -1.0 # 气温骤降判据
# ===============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 字体设置 ---
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

# --- 模型定义 ---

# 1. 标准 Wood-Kwok (仅用于筛选阶段的"体检")
def model_wk_std(z, u_max, delta):
    z = np.maximum(z, 0.1)
    # alpha 固定为 1/6
    return 1.55 * u_max * np.power(z/delta, 1.0/6.0) * (1 - erf(0.7*z/delta))

# 2. 广义 Wood-Kwok (用于最终拟合，Alpha 可变)
def model_wk_custom(z, u_max, delta, alpha):
    z = np.maximum(z, 0.1)
    # alpha 是变量，反映地面粗糙度
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

# --- 核心处理 ---
def process_station(file_path):
    station_code = get_station_code(file_path)
    print(f"[{station_code}] 分析中...", end=' ')

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
    mat_mean = np.full((n, m), np.nan)

    for i, h in enumerate(heights):
        c_g = next((c for c in df.columns if f'{h}m' in c and ('最大' in c or 'Gust' in c) and is_clean_col(c)), None)
        c_m = next((c for c in df.columns if f'{h}m' in c and ('水平' in c or 'Speed' in c) and is_clean_col(c)), None)
        if not c_g and c_m: c_g = c_m
        
        if c_g: mat_gust[:, i] = pd.to_numeric(df[c_g], errors='coerce').values
        if c_m: mat_mean[:, i] = pd.to_numeric(df[c_m], errors='coerce').values

    # 3. 筛选循环 (收集面条)
    valid_profiles = [] # 存储所有通过筛选的原始廓线
    months = df['Date/Time'].dt.month.values
    hours = df['Date/Time'].dt.hour.values

    for i in range(n):
        # 物理环境筛选
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
        
        # 形状简单校验
        if not (row_g[idx_max] > row_g[0]+0.5 and row_g[idx_max] > row_g[-1]+0.5): continue

        # 【关键】用标准模型(1/6)进行"体检"
        try:
            mask = ~np.isnan(row_g)
            if np.sum(mask) < 4: continue
            # 先用标准尺子量，剔除明显的 LLJ
            p_check, _ = curve_fit(model_wk_std, np.array(heights)[mask], row_g[mask], 
                                   p0=[u_max, 300], bounds=([5,10], [100,2500]))
            
            if p_check[1] > MAX_ALLOWED_DELTA: continue
            
            # 通过体检，加入面条集合
            valid_profiles.append(row_g)
        except: continue

    count = len(valid_profiles)
    print(f"N={count}", end=' ')
    if count < 1: 
        print("(无有效样本)")
        return None

    # 4. 计算平均并进行"定制化拟合"
    valid_profiles = np.array(valid_profiles)
    avg_profile = np.nanmean(valid_profiles, axis=0)

    # 拟合广义模型 (3参数: U, Delta, Alpha)
    try:
        popt, pcov = curve_fit(model_wk_custom, heights, avg_profile, 
                               p0=[15, 300, 0.16], 
                               bounds=([5, 10, 0.05], [100, 2500, 0.45]))
        
        final_u = popt[0]
        final_delta = popt[1]
        final_alpha = popt[2]
        
        # R2
        ss_res = np.sum((avg_profile - model_wk_custom(heights, *popt))**2)
        ss_tot = np.sum((avg_profile - np.mean(avg_profile))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"| Alpha={final_alpha:.3f}")
    except:
        print("拟合失败")
        return None

    # 5. 绘图 (全家福)
    plt.figure(figsize=(8, 10))
    
    # A. 画所有单体事件 (灰色面条)
    for prof in valid_profiles:
        #plt.plot(prof, heights, color='gray', alpha=0.25, linewidth=0.8)
        # 修改为：颜色用深灰，不透明度调高到 0.4，线宽加到 1.0
        plt.plot(prof, heights, color='dimgray', alpha=0.4, linewidth=1.0)
        
    # B. 画平均观测点 (黑色散点)
    plt.plot(avg_profile, heights, 'ko', label='Observed Mean', markersize=6, zorder=5)
    
    # C. 画定制化拟合曲线 (红色粗线)
    z_smooth = np.linspace(0, heights[-1]*1.1, 100)
    y_custom = model_wk_custom(z_smooth, final_u, final_delta, final_alpha)
    plt.plot(y_custom, z_smooth, 'r-', linewidth=3, label=f'Custom Model ($\\alpha={final_alpha:.3f}$)', zorder=10)
    
    # D. 画标准模型对比 (蓝色虚线，可选) - 展示你的优化有多重要
    # 为了对比公平，保持Umax和Delta一致，只把Alpha切回1/6
    y_std = model_wk_custom(z_smooth, final_u, final_delta, 1.0/6.0)
    plt.plot(y_std, z_smooth, 'b--', linewidth=2, alpha=0.7, label='Standard WK ($\\alpha=1/6$)', zorder=9)

    # 装饰
    plt.title(f"{station_code} Customized Downburst Profile\nShape Parameter $\\alpha={final_alpha:.3f}$ (N={count})", fontsize=14)
    plt.xlabel('Gust Speed [m/s]', fontsize=12)
    plt.ylabel('Height [m]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 信息框
    info_text = (
        f"Fitted Parameters:\n"
        f"  $U_{{max}}$ = {final_u:.1f} m/s\n"
        f"  $\delta$ = {final_delta:.0f} m\n"
        f"  $\\alpha$ = {final_alpha:.3f}\n"
        f"  $R^2$ = {r2:.2f}"
    )
    plt.gca().text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{station_code}_Custom_Spaghetti.png'), dpi=300)
    plt.close()

    return {
        'Station': station_code,
        'Count': count,
        'Param_Umax': final_u,
        'Param_Delta': final_delta,
        'Param_Alpha': final_alpha,
        'R2': r2
    }

# --- 主程序 ---
def main():
    print("="*60)
    print(" 下击暴流全景建模 (Spaghetti + Custom Alpha)")
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
        df.to_excel(os.path.join(OUTPUT_DIR, 'Downburst_Final_Params.xlsx'), index=False)
        print(f"\n全部完成！请查看 {OUTPUT_DIR} 下的图表。")

if __name__ == "__main__":
    main()