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
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/downburst_analysis_final'

# 筛选标准
MIN_GUST_SPEED = 12.0      
MIN_GUST_FACTOR = 1.25     # 保持适当宽松，靠后面的拟合来剔除

# 【关键】最大允许的射流厚度 delta
# 对应鼻尖高度约 180m。超过这个高度的一律视为 LLJ 剔除。
MAX_ALLOWED_DELTA = 800.0  

PLOT_EVENTS = True
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
if PLOT_EVENTS:
    os.makedirs(os.path.join(OUTPUT_DIR, 'events'), exist_ok=True)
warnings.filterwarnings('ignore')

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

def model_wood_kwok(z, u_max, delta):
    z = np.maximum(z, 0.1)
    term1 = 1.55 * u_max
    term2 = np.power(z / delta, 1.0/6.0)
    term3 = 1.0 - erf(0.7 * z / delta)
    return term1 * term2 * term3

def is_clean_col(col_name):
    blacklist = ['最小', '偏差', '矢量', '标准差', 'Min', 'Std', 'Dev', 'Vector', 'Sigma']
    for bad in blacklist:
        if bad.lower() in col_name.lower(): return False
    return True

def get_col_by_type(columns, height, type_keywords):
    for c in columns:
        if f'{height}m' not in c: continue
        if not any(k in c for k in type_keywords): continue
        if is_clean_col(c): return c
    return None

def strict_tab_parse(file_path):
    encodings = ['utf-8', 'gbk', 'gb18030', 'latin-1']
    for enc in encodings:
        try:
            temp = pd.read_csv(file_path, sep='\t', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp.columns) or 'Speed' in str(temp.columns): return temp
            temp = pd.read_csv(file_path, sep='\s+', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp.columns): return temp
        except: continue
    return None

def analyze_station_downburst(file_path):
    filename = os.path.basename(file_path)
    try:
        station_name = filename.split('-')[0] if SYSTEM_ZH_FONT else filename.split('-')[1]
    except: station_name = filename[:10]
    
    print(f"[{station_name}] 处理中...", end=' ')
    df = strict_tab_parse(file_path)
    if df is None: return None
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    max_ws_cols_temp = [c for c in df.columns if ('最大' in c or 'Gust' in c or 'Max' in c) and is_clean_col(c) and 'm' in c]
    use_mean_as_max = False
    if not max_ws_cols_temp:
        max_ws_cols_temp = [c for c in df.columns if ('水平风速' in c or 'Speed' in c) and is_clean_col(c)]
        use_mean_as_max = True

    heights = sorted(list(set([int(re.search(r'(\d+)m', c).group(1)) for c in max_ws_cols_temp if re.search(r'(\d+)m', c)])))
    if len(heights) < 4: return None

    n_samples = len(df)
    n_heights = len(heights)
    
    mat_gust = np.full((n_samples, n_heights), np.nan)
    mat_mean = np.full((n_samples, n_heights), np.nan)
    
    for i, h in enumerate(heights):
        c_g = get_col_by_type(df.columns, h, ['最大', 'Gust', 'Max'])
        if not c_g: c_g = get_col_by_type(df.columns, h, ['水平', 'Speed'])
        c_m = get_col_by_type(df.columns, h, ['水平', 'Speed'])
        
        if c_g: mat_gust[:, i] = pd.to_numeric(df[c_g], errors='coerce').values
        if c_m: mat_mean[:, i] = pd.to_numeric(df[c_m], errors='coerce').values

    candidates = []
    for i in range(n_samples):
        row_gust = mat_gust[i, :]
        row_mean = mat_mean[i, :]
        if np.isnan(row_gust).all(): continue
        
        u_max_val = np.nanmax(row_gust)
        if u_max_val < MIN_GUST_SPEED: continue
        
        idx_max = np.nanargmax(row_gust)
        u_mean_val = row_mean[idx_max]
        gf = u_max_val / u_mean_val if u_mean_val > 1 else 1.0
        
        if not use_mean_as_max and gf < MIN_GUST_FACTOR: continue
        
        z_max = heights[idx_max]
        if idx_max == 0 or idx_max == n_heights - 1: continue 
        
        # 简单形状判定: 必须是凸起的
        if not (row_gust[idx_max] > row_gust[0] + 1.0 and row_gust[idx_max] > row_gust[-1] + 1.0):
            continue
            
        candidates.append(i)

    # 拟合与强力剔除
    fitted_params = []
    r2_scores = []
    final_count = 0
    
    for idx in candidates:
        y_data = mat_gust[idx, :]
        x_data = np.array(heights)
        mask = ~np.isnan(y_data)
        x_valid = x_data[mask]
        y_valid = y_data[mask]
        
        if len(y_valid) < 4: continue
        u_max_obs = np.max(y_valid)
        
        try:
            # 拟合
            popt, _ = curve_fit(model_wood_kwok, x_valid, y_valid, 
                                p0=[u_max_obs, 300], bounds=([5, 10], [100, 2500]))
            
            # 【核心改进】: 强力剔除高鼻尖 (疑似LLJ)
            if popt[1] > MAX_ALLOWED_DELTA: 
                continue 

            # 计算 R2
            y_pred = model_wood_kwok(x_valid, *popt)
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            if r2 > 0.6: # 门槛适中
                fitted_params.append(popt)
                r2_scores.append(r2)
                final_count += 1
                
                # 只有通过了双重考验的才画图
                if final_count <= 5 and PLOT_EVENTS:
                    timestamp = df.iloc[idx]['Date/Time'].replace(':','-').replace(' ','_')
                    plt.figure(figsize=(5, 6))
                    plt.plot(y_valid, x_valid, 'ko', label='Obs')
                    z_smooth = np.linspace(0, heights[-1], 100)
                    plt.plot(model_wood_kwok(z_smooth, *popt), z_smooth, 'r-', lw=2, 
                             label=f'δ={popt[1]:.0f}m')
                    plt.title(f"{station_name}\nReal Downburst?\nZ_nose approx {popt[1]*0.23:.0f}m")
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(os.path.join(OUTPUT_DIR, 'events', f'{station_name}_{timestamp}.png'))
                    plt.close()

        except: continue

    print(f"| 最终确认为下击暴流: {len(fitted_params)} 个")
    
    if not fitted_params: return None
    
    avg_params = np.mean(fitted_params, axis=0)
    return {
        'Station': station_name,
        'Confirmed_Count': len(fitted_params),
        'WK_Umax_Avg': avg_params[0],
        'WK_Delta_Avg': avg_params[1],
        'Avg_R2': np.mean(r2_scores)
    }

def main():
    print("="*60)
    print(" 下击暴流终极筛选 (The Final Filter)")
    print(f" 剔除标准: 射流厚度 > {MAX_ALLOWED_DELTA}m (即鼻尖高度 > {MAX_ALLOWED_DELTA*0.23:.0f}m)")
    print("="*60)
    
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    results = []
    
    for f in files:
        res = analyze_station_downburst(f)
        if res: results.append(res)
        
    if results:
        df_res = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, 'Downburst_Final_Results.xlsx')
        df_res.to_excel(out_path, index=False)
        print("\n最终结果表已生成，请查看 Excel。")
        print(df_res[['Station', 'Confirmed_Count', 'WK_Delta_Avg']])

if __name__ == "__main__":
    main()