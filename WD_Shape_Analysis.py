import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
from sklearn.metrics import mean_squared_error

# ================= 配置区域 =================
# 替换为你的文件路径
FILE_PATH = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/wd_shape_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 质量控制
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

# --- 1. 定义模型 ---

def model_linear(z, k, b):
    """线性模型: WD = k*z + b"""
    return k * z + b

def model_quadratic(z, a, b, c):
    """二次模型: WD = a*z^2 + b*z + c (模拟 Ekman 螺旋的曲率)"""
    return a * z**2 + b * z + c

# --- 2. 核心工具 ---

def unwrap_deg(degrees):
    """
    关键函数：解缠绕
    将 [350, 355, 5, 10] 这种跨越 0 度的数据
    转换为 [350, 355, 365, 370] 这种连续数据，以便拟合
    """
    rads = np.radians(degrees)
    unwrapped_rads = np.unwrap(rads)
    return np.degrees(unwrapped_rads)

def vector_mean(wd_array):
    """计算一组风向的矢量平均"""
    rads = np.radians(wd_array)
    u = np.nanmean(np.sin(rads))
    v = np.nanmean(np.cos(rads))
    deg = np.degrees(np.arctan2(u, v))
    if deg < 0: deg += 360
    return deg

def strict_tab_parse_v3(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'latin-1']
    raw_lines = []
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f: raw_lines = f.readlines()
            break
        except: continue
    if not raw_lines: return None
    header_idx = -1
    for i, line in enumerate(raw_lines[:100]):
        if "m水平风速" in line: header_idx = i; break
    if header_idx == -1: return None
    header = raw_lines[header_idx].strip().split('\t')
    header = [h.strip().replace('"', '') for h in header]
    data = []
    for i in range(header_idx + 1, len(raw_lines)):
        line = raw_lines[i].strip()
        if not line: continue
        parts = line.split('\t')
        parts = [p.strip().replace('"', '') for p in parts]
        if len(parts) > len(header): parts = parts[:len(header)]
        elif len(parts) < len(header): parts += [''] * (len(header) - len(parts))
        data.append(parts)
    return pd.DataFrame(data, columns=header)

# --- 3. 分析逻辑 ---

def analyze_wd_shape(file_path):
    print(f"正在分析文件: {os.path.basename(file_path)}")
    df_raw = strict_tab_parse_v3(file_path)
    if df_raw is None: return

    # 提取列
    df_raw.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df_raw.columns]
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    print(" -> 正在提取风向数据并筛选急流...")
    events = []
    
    # 为了加速，先把列名都找出来
    wd_cols_map = {}
    ws_cols_map = {}
    for h in heights:
        ws_c = f'{h}m水平风速'
        # 模糊匹配风向列
        wd_c = next((c for c in df_raw.columns if str(h) in c and ('风向' in c or 'Direction' in c) and '最大' not in c), None)
        if ws_c in df_raw.columns and wd_c:
            ws_cols_map[h] = ws_c
            wd_cols_map[h] = wd_c

    # 逐行处理
    for idx in df_raw.index:
        # 1. 检查是否 LLJ
        # 先快速提取风速判断
        ws_vals = []
        valid_h = []
        
        for h in heights:
            if h in ws_cols_map:
                try:
                    val = float(df_raw.loc[idx, ws_cols_map[h]])
                    ws_vals.append(val)
                    valid_h.append(h)
                except: pass
        
        if not ws_vals: continue
        ws_arr = np.array(ws_vals)
        
        mx_i = np.argmax(ws_arr)
        mx_h = valid_h[mx_i]
        
        # LLJ 判定
        if (mx_h > MIN_JET_HEIGHT) and (mx_h < MAX_JET_HEIGHT):
            if (ws_arr[mx_i] - ws_arr[0] >= LLJ_THRESHOLD) and (ws_arr[mx_i] - ws_arr[-1] >= LLJ_THRESHOLD):
                # 是急流，提取风向
                wd_vals = []
                final_h = []
                for h in valid_h:
                    try:
                        d = float(df_raw.loc[idx, wd_cols_map[h]])
                        if not np.isnan(d):
                            wd_vals.append(d)
                            final_h.append(h)
                    except: pass
                
                if len(wd_vals) > 5:
                    events.append({
                        'z': np.array(final_h),
                        'wd': np.array(wd_vals)
                    })

    print(f" -> 捕获 LLJ 样本: {len(events)} 个")
    if len(events) < 10: return

    # --- Step 1: 竞技场 (Per Event Fitting) ---
    print("\n[阶段 1] 模型竞技场 (Per-Event Comparison)...")
    errors = {'Linear': [], 'Quadratic': []}
    
    for ev in events:
        z = ev['z']
        wd_raw = ev['wd']
        
        # 关键步骤：解缠绕 (Unwrap)
        # 这一步把 350, 10 变成 350, 370，使其连续
        wd_cont = unwrap_deg(wd_raw)
        
        # 1. Linear Fit
        try:
            popt, _ = curve_fit(model_linear, z, wd_cont)
            wd_pred = model_linear(z, *popt)
            rmse = np.sqrt(mean_squared_error(wd_cont, wd_pred))
            errors['Linear'].append(rmse)
        except: pass
        
        # 2. Quadratic Fit
        try:
            popt, _ = curve_fit(model_quadratic, z, wd_cont)
            wd_pred = model_quadratic(z, *popt)
            rmse = np.sqrt(mean_squared_error(wd_cont, wd_pred))
            errors['Quadratic'].append(rmse)
        except: pass

    # 打印结果
    print("-" * 40)
    best_model_name = "Linear"
    min_median = 999
    
    for name, errs in errors.items():
        if not errs: continue
        med = np.median(errs)
        mean_err = np.mean(errs)
        print(f"Model: {name:10s} | Median RMSE: {med:.3f} deg | Mean: {mean_err:.3f}")
        if med < min_median:
            min_median = med
            best_model_name = name
    print("-" * 40)
    print(f"🏆 冠军模型: {best_model_name}")

    # --- Step 2: 全局平均廓线分析 ---
    print("\n[阶段 2] 全局平均廓线拟合...")
    
    # 计算矢量平均廓线 (Vector Mean Profile)
    # 因为每个事件的高度层可能略有不同(缺失值)，这里取最全的高度层
    common_heights = heights
    mean_wd_profile = []
    
    for h in common_heights:
        vals_at_h = []
        for ev in events:
            # 找该事件中对应高度的值
            if h in ev['z']:
                idx = np.where(ev['z'] == h)[0][0]
                vals_at_h.append(ev['wd'][idx])
        
        if vals_at_h:
            mean_wd_profile.append(vector_mean(np.array(vals_at_h)))
        else:
            mean_wd_profile.append(np.nan)
            
    # 清洗 NaN
    mean_wd_profile = np.array(mean_wd_profile)
    mask = ~np.isnan(mean_wd_profile)
    z_fit = np.array(common_heights)[mask]
    wd_fit = mean_wd_profile[mask]
    
    # 解缠绕平均廓线
    wd_fit_cont = unwrap_deg(wd_fit)
    
    # 拟合最优模型
    if best_model_name == "Linear":
        popt_best, _ = curve_fit(model_linear, z_fit, wd_fit_cont)
        formula_str = f"WD(z) = {popt_best[0]:.4f} * z + {popt_best[1]:.2f}"
        wd_pred_plot = model_linear(z_fit, *popt_best)
    else:
        popt_best, _ = curve_fit(model_quadratic, z_fit, wd_fit_cont)
        formula_str = f"WD(z) = {popt_best[0]:.2e} * z^2 + {popt_best[1]:.4f} * z + {popt_best[2]:.2f}"
        wd_pred_plot = model_quadratic(z_fit, *popt_best)
        
    print(f"\n>>> 推荐通用公式 ({best_model_name}):")
    print(f"    {formula_str}")
    
    # --- Step 3: 绘图 ---
    plt.figure(figsize=(9, 7))
    
    # 画原始平均点 (注意：为了画图美观，把解缠绕后的值画出来，否则会有断层)
    plt.plot(wd_fit_cont, z_fit, 'ko', markersize=8, label='Vector Mean WD (Unwrapped)')
    
    # 画拟合线
    z_smooth = np.linspace(z_fit[0], z_fit[-1], 200)
    if best_model_name == "Linear":
        y_smooth = model_linear(z_smooth, *popt_best)
    else:
        y_smooth = model_quadratic(z_smooth, *popt_best)
        
    plt.plot(y_smooth, z_smooth, 'b-', linewidth=2, label=f'{best_model_name} Fit')
    
    # 标注公式
    plt.title(f'Wind Direction Profile Shape Analysis\nWinner: {best_model_name}', fontsize=14)
    plt.xlabel('Wind Direction [deg] (Continuous)', fontsize=12)
    plt.ylabel('Height [m]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 在图上写公式
    plt.text(0.05, 0.95, f"Formula:\n{formula_str}", transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    out_img = os.path.join(OUTPUT_DIR, 'WD_Shape_Fit.png')
    plt.savefig(out_img, dpi=300)
    print(f"[图表] 拟合图已保存: {out_img}")

if __name__ == "__main__":
    analyze_wd_shape(FILE_PATH)