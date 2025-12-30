import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy.special import erf  # Wood-Kwok 模型需要误差函数
import re
import os
import glob
import warnings

# ================= 用户配置区域 =================
# 数据路径
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/downburst_analysis'

# 下击暴流筛选标准 (严苛模式)
MIN_GUST_SPEED = 12.0      # 1. 最大阵风必须 > 12m/s (你可以调高到15-18)
MIN_GUST_FACTOR = 1.3      # 2. 阵风系数 (Gust/Mean) > 1.3
TEMP_DROP_THRESHOLD = 1.0  # 3. 30分钟内气温下降 > 1.0度 (冷池效应)
# 注：如果找不到气温数据，代码会自动降级为只按风速筛选

# 拟合配置
PLOT_EVENTS = True         # 是否为每个事件画图
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
if PLOT_EVENTS:
    os.makedirs(os.path.join(OUTPUT_DIR, 'events'), exist_ok=True)
warnings.filterwarnings('ignore')

# --- 0. 字体配置 ---
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

# --- 1. 下击暴流专用模型 (Wood-Kwok) ---
def model_wood_kwok(z, u_max, delta):
    """
    Wood and Kwok (1998) 下击暴流经验模型
    U(z) = 1.55 * U_max * (z/delta)^(1/6) * [1 - erf(0.7 * z/delta)]
    
    参数:
      u_max: 廓线中的最大风速值
      delta: 射流厚度参数 (风速减小到一半的高度附近)
    """
    # 避免 z=0 或负数
    z = np.maximum(z, 0.1)
    
    term1 = 1.55 * u_max
    term2 = np.power(z / delta, 1.0/6.0)
    term3 = 1.0 - erf(0.7 * z / delta)
    
    return term1 * term2 * term3

# --- 2. 增强型数据读取 (找最大风速 & 外温) ---
def is_clean_col(col_name):
    # 排除不需要的统计量，但这次我们要保留 '最大'
    blacklist = ['最小', '偏差', '矢量', '标准差', 'Min', 'Std', 'Dev', 'Vector', 'Sigma']
    for bad in blacklist:
        if bad.lower() in col_name.lower(): return False
    return True

def get_col_by_type(columns, height, type_keywords):
    """根据高度和关键字找列"""
    for c in columns:
        if f'{height}m' not in c: continue
        # 必须包含关键字之一
        if not any(k in c for k in type_keywords): continue
        if is_clean_col(c): return c
    return None

def find_temp_col(columns):
    """全局搜索气温列 (外温/Temperature)"""
    for c in columns:
        if '外温' in c or 'Temperature' in c or 'T_10m' in c:
            return c
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

# --- 3. 核心分析逻辑 ---
def analyze_station_downburst(file_path):
    filename = os.path.basename(file_path)
    try:
        station_name = filename.split('-')[0] if SYSTEM_ZH_FONT else filename.split('-')[1]
    except: station_name = filename[:10]
    
    print(f"[{station_name}] 读取中...", end=' ')
    df = strict_tab_parse(file_path)
    if df is None: return None
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    # --- 1. 识别高度 & 关键列 ---
    # 找“最大风速”列 (Max/Gust)
    max_ws_cols_temp = [c for c in df.columns if ('最大' in c or 'Gust' in c or 'Max' in c) and is_clean_col(c) and 'm' in c]
    # 如果没找到最大风速，只能退而求其次用水平风速，但会警告
    use_mean_as_max = False
    if not max_ws_cols_temp:
        # print("(未找到最大风速列，使用平均风速替代，效果可能打折)")
        max_ws_cols_temp = [c for c in df.columns if ('水平风速' in c or 'Speed' in c) and is_clean_col(c)]
        use_mean_as_max = True

    heights = sorted(list(set([int(re.search(r'(\d+)m', c).group(1)) for c in max_ws_cols_temp if re.search(r'(\d+)m', c)])))
    
    if len(heights) < 4: 
        print("高度层不足")
        return None

    # 构建矩阵
    n_samples = len(df)
    n_heights = len(heights)
    
    mat_gust = np.full((n_samples, n_heights), np.nan) # 最大风速
    mat_mean = np.full((n_samples, n_heights), np.nan) # 平均风速 (用于算阵风系数)
    
    for i, h in enumerate(heights):
        # 1. 找 Gust 列
        c_g = get_col_by_type(df.columns, h, ['最大', 'Gust', 'Max'])
        if not c_g: c_g = get_col_by_type(df.columns, h, ['水平', 'Speed']) # Fallback
        
        # 2. 找 Mean 列
        c_m = get_col_by_type(df.columns, h, ['水平', 'Speed'])
        
        if c_g: mat_gust[:, i] = pd.to_numeric(df[c_g], errors='coerce').values
        if c_m: mat_mean[:, i] = pd.to_numeric(df[c_m], errors='coerce').values

    # 获取气温数据
    col_temp = find_temp_col(df.columns)
    arr_temp = np.full(n_samples, np.nan)
    if col_temp:
        arr_temp = pd.to_numeric(df[col_temp], errors='coerce').values
        # 计算 30分钟温差 (假设数据是10min间隔，即前移3格)
        # Delta T = T_now - T_past (下击暴流通常是负值，突降)
        temp_series = pd.Series(arr_temp)
        arr_temp_diff = temp_series.diff(3).values # 3个步长
    else:
        arr_temp_diff = np.zeros(n_samples) # 没气温就不筛选这一项

    # --- 2. 下击暴流筛选 ---
    candidates = []
    
    # 取最高层的风速作为参考 (或者取最大值所在层)
    # 这里我们遍历每个时刻
    for i in range(3, n_samples): # 从3开始因为要算温差
        # 基础数据检查
        row_gust = mat_gust[i, :]
        row_mean = mat_mean[i, :]
        if np.isnan(row_gust).all(): continue
        
        # A. 强度判定
        u_max_val = np.nanmax(row_gust)
        if u_max_val < MIN_GUST_SPEED: continue
        
        # B. 阵风系数判定 (Gust Factor)
        # 找最大风速对应的高度索引
        idx_max = np.nanargmax(row_gust)
        u_mean_val = row_mean[idx_max]
        
        if u_mean_val < 1.0: continue # 避免除零
        gf = u_max_val / u_mean_val
        
        if not use_mean_as_max and gf < MIN_GUST_FACTOR: continue
        
        # C. 气温突降判定 (Cold Pool)
        # 如果有气温数据，且温差 > 阈值 (注意是下降，所以 diff < -threshold)
        if col_temp and (arr_temp_diff[i] > -TEMP_DROP_THRESHOLD): 
            # 如果温差不够大，可能不是下击暴流，或者是干微暴流
            # 这里为了严格，可以continue，或者放宽
            # 暂时放宽：如果有气温数据但没降温，可能是 LLJ，跳过
            continue

        # D. 廓线形状判定 (必须有鼻尖，且鼻尖不能太高)
        # 下击暴流鼻尖通常 < 150m
        z_max = heights[idx_max]
        if idx_max == 0 or idx_max == n_heights - 1: continue # 贴地或最高处不算
        # 简单检查单峰: 比上下都大
        if not (row_gust[idx_max] > row_gust[0] + 1.0 and row_gust[idx_max] > row_gust[-1] + 1.0):
            continue
            
        # 通过所有测试
        candidates.append(i)

    print(f"筛选出 {len(candidates)} 个疑似个例", end=' ')
    if not candidates: 
        print("")
        return None

    # --- 3. 拟合与统计 ---
    fitted_params = []
    r2_scores = []
    
    # 随机抽一些画图 (如果太多)
    plot_indices = candidates[:5] if len(candidates) > 5 else candidates
    
    for idx in candidates:
        y_data = mat_gust[idx, :] # x是风速
        x_data = np.array(heights) # y是高度
        
        # 清洗 NaN
        mask = ~np.isnan(y_data)
        x_valid = x_data[mask]
        y_valid = y_data[mask]
        
        if len(y_valid) < 4: continue
        
        u_max_obs = np.max(y_valid)
        
        try:
            # Wood-Kwok 拟合
            # p0: [u_max, delta]
            # u_max 猜观测最大值，delta 猜 300m
            popt, _ = curve_fit(model_wood_kwok, x_valid, y_valid, 
                                p0=[u_max_obs, 300], 
                                bounds=([5, 10], [100, 2000]))
            
            # 计算 R2
            y_pred = model_wood_kwok(x_valid, *popt)
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            if r2 > 0.8: # 只保留拟合好的
                fitted_params.append(popt)
                r2_scores.append(r2)
                
                # 画图 (仅限前几个)
                if idx in plot_indices and PLOT_EVENTS:
                    timestamp = df.iloc[idx]['Date/Time'].replace(':','-').replace(' ','_')
                    plt.figure(figsize=(6, 8))
                    plt.plot(y_valid, x_valid, 'ko', label='Gust Obs')
                    
                    z_smooth = np.linspace(0, heights[-1], 100)
                    plt.plot(model_wood_kwok(z_smooth, *popt), z_smooth, 'r-', lw=2, 
                             label=f'Wood-Kwok\nUmax={popt[0]:.1f}, δ={popt[1]:.0f}')
                    
                    plt.title(f"Downburst Event: {station_name}\nTime: {timestamp} | TempDrop: {arr_temp_diff[idx]:.1f}C")
                    plt.xlabel('Gust Speed [m/s]')
                    plt.ylabel('Height [m]')
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(os.path.join(OUTPUT_DIR, 'events', f'{station_name}_{timestamp}.png'))
                    plt.close()

        except: continue

    print(f"| 成功拟合 {len(fitted_params)} 个")
    
    if not fitted_params: return None
    
    # 返回平均参数
    avg_params = np.mean(fitted_params, axis=0)
    return {
        'Station': station_name,
        'Count': len(fitted_params),
        'WK_Umax_Avg': avg_params[0],
        'WK_Delta_Avg': avg_params[1], # 射流厚度
        'Avg_R2': np.mean(r2_scores)
    }

# --- 4. 主程序 ---
def main():
    print("="*60)
    print(" 下击暴流 (Downburst) 专用筛选与拟合程序")
    print(" 模型: Wood and Kwok (1998)")
    print("="*60)
    
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    results = []
    
    for f in files:
        res = analyze_station_downburst(f)
        if res: results.append(res)
        
    if results:
        df_res = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, 'Downburst_Params_Summary.xlsx')
        df_res.to_excel(out_path, index=False)
        print("\n" + "="*60)
        print(f"汇总结果已保存: {out_path}")
        print(df_res)
    else:
        print("\n未发现明显的下击暴流事件 (可能是阈值太高或数据中没有)。")

if __name__ == "__main__":
    main()