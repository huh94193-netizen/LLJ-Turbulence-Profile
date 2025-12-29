import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
from sklearn.metrics import mean_squared_error

# ================= 配置区域 =================
FILE_PATH = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/wd_shape_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

# --- 1. 定义相对模型 (Relative Models) ---
# 注意：这里不再拟合截距 b，因为截距被强制设为 0 (相对于底层)

def model_linear_relative(delta_z, rate):
    """
    线性偏转模型
    Delta_WD = rate * Delta_Z
    """
    return rate * delta_z

def model_quadratic_relative(delta_z, rate, curve):
    """
    二次偏转模型 (Ekman螺旋)
    Delta_WD = rate * Delta_Z + curve * (Delta_Z)^2
    """
    return rate * delta_z + curve * delta_z**2

# --- 2. 核心工具 ---
def unwrap_deg(degrees):
    """解缠绕：解决 350->10 的突变问题"""
    rads = np.radians(degrees)
    unwrapped_rads = np.unwrap(rads)
    return np.degrees(unwrapped_rads)

def strict_tab_parse_v3(file_path):
    # (保持原有的读取逻辑)
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
def analyze_wd_relative(file_path):
    print(f"正在分析文件 (相对风向模式): {os.path.basename(file_path)}")
    df_raw = strict_tab_parse_v3(file_path)
    if df_raw is None: return

    # 提取列
    df_raw.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df_raw.columns]
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    # 映射列名
    wd_cols_map = {}
    ws_cols_map = {}
    for h in heights:
        ws_c = f'{h}m水平风速'
        wd_c = next((c for c in df_raw.columns if str(h) in c and ('风向' in c or 'Direction' in c) and '最大' not in c), None)
        if ws_c in df_raw.columns and wd_c:
            ws_cols_map[h] = ws_c
            wd_cols_map[h] = wd_c

    # 提取事件
    events = []
    print(" -> 正在提取并计算相对转角 (Delta WD)...")
    
    for idx in df_raw.index:
        # 1. 提取风速判断 LLJ
        ws_vals, valid_h_ws = [], []
        for h in heights:
            if h in ws_cols_map:
                try:
                    v = float(df_raw.loc[idx, ws_cols_map[h]])
                    ws_vals.append(v)
                    valid_h_ws.append(h)
                except: pass
        
        if not ws_vals: continue
        ws_arr = np.array(ws_vals)
        mx_i = np.argmax(ws_arr)
        mx_h = valid_h_ws[mx_i]
        
        # LLJ 判定
        if (mx_h > MIN_JET_HEIGHT) and (mx_h < MAX_JET_HEIGHT):
            if (ws_arr[mx_i] - ws_arr[0] >= LLJ_THRESHOLD) and (ws_arr[mx_i] - ws_arr[-1] >= LLJ_THRESHOLD):
                # 2. 提取风向
                wd_vals, valid_h_wd = [], []
                for h in heights: # 确保高度有序
                    if h in wd_cols_map:
                        try:
                            d = float(df_raw.loc[idx, wd_cols_map[h]])
                            if not np.isnan(d):
                                wd_vals.append(d)
                                valid_h_wd.append(h)
                        except: pass
                
                if len(wd_vals) > 5:
                    z = np.array(valid_h_wd)
                    wd = np.array(wd_vals)
                    
                    # 关键处理：计算相对于底层的转角
                    # 1. 解缠绕
                    wd_cont = unwrap_deg(wd)
                    # 2. 归零 (Subtract Base)
                    wd_base = wd_cont[0]
                    z_base = z[0]
                    
                    delta_wd = wd_cont - wd_base
                    delta_z = z - z_base
                    
                    events.append({
                        'delta_z': delta_z,
                        'delta_wd': delta_wd,
                        'wd_base': wd_base, # 记录下来备查
                        'z_base': z_base
                    })

    print(f" -> 捕获 LLJ 样本: {len(events)} 个")
    if len(events) < 10: return

    # --- Step 1: 竞技场 (拟合 Delta WD) ---
    print("\n[阶段 1] 相对转角模型竞技...")
    errors = {'Linear_Rate': [], 'Quadratic_Rate': []}
    
    # 收集所有点用于画总图
    all_dz = []
    all_dwd = []
    
    for ev in events:
        dz = ev['delta_z']
        dwd = ev['delta_wd']
        
        all_dz.extend(dz)
        all_dwd.extend(dwd)
        
        # 1. Linear Fit (过原点)
        try:
            popt, _ = curve_fit(model_linear_relative, dz, dwd)
            pred = model_linear_relative(dz, *popt)
            rmse = np.sqrt(mean_squared_error(dwd, pred))
            errors['Linear_Rate'].append(rmse)
        except: pass
        
        # 2. Quadratic Fit (过原点)
        try:
            popt, _ = curve_fit(model_quadratic_relative, dz, dwd)
            pred = model_quadratic_relative(dz, *popt)
            rmse = np.sqrt(mean_squared_error(dwd, pred))
            errors['Quadratic_Rate'].append(rmse)
        except: pass

    # 打印结果
    print("-" * 40)
    best_model = "Linear_Rate"
    min_median = 999
    
    for name, errs in errors.items():
        if not errs: continue
        med = np.median(errs)
        print(f"Model: {name:15s} | Median RMSE: {med:.3f} deg")
        if med < min_median:
            min_median = med
            best_model = name
    print("-" * 40)
    print(f"🏆 冠军模型: {best_model}")

    # --- Step 2: 拟合通用参数 ---
    print("\n[阶段 2] 计算通用偏转参数...")
    all_dz = np.array(all_dz)
    all_dwd = np.array(all_dwd)
    
    # 过滤极端值以便绘图好看
    mask = (np.abs(all_dwd) < 90) # 偏转超过90度的很少见，可能是坏数
    all_dz_clean = all_dz[mask]
    all_dwd_clean = all_dwd[mask]
    
    z_smooth = np.linspace(0, np.max(all_dz_clean), 100)
    
    # 拟合最优曲线
    if best_model == "Linear_Rate":
        popt_best, _ = curve_fit(model_linear_relative, all_dz_clean, all_dwd_clean)
        formula_str = f"WD(z) = WD_base + {popt_best[0]:.4f} * (z - z_base)"
        y_smooth = model_linear_relative(z_smooth, *popt_best)
        print(f"  -> 平均偏转率 (Veering Rate): {popt_best[0]:.4f} deg/m")
        if popt_best[0] > 0: print("     (顺时针偏转 / Veering)")
        else: print("     (逆时针偏转 / Backing)")
    else:
        popt_best, _ = curve_fit(model_quadratic_relative, all_dz_clean, all_dwd_clean)
        formula_str = f"WD(z) = WD_base + {popt_best[0]:.2e}*(z-z_b) + {popt_best[1]:.2e}*(z-z_b)^2"
        y_smooth = model_quadratic_relative(z_smooth, *popt_best)
        print(f"  -> 线性项系数: {popt_best[0]:.2e}")
        print(f"  -> 二次项系数: {popt_best[1]:.2e}")

    # --- Step 3: 绘图 ---
    plt.figure(figsize=(10, 8))
    
    # 画背景散点 (降采样，避免卡顿)
    if len(all_dz_clean) > 5000:
        idx = np.random.choice(len(all_dz_clean), 5000, replace=False)
        plt.scatter(all_dwd_clean[idx], all_dz_clean[idx], color='gray', s=1, alpha=0.1, label='Relative WD Samples')
    else:
        plt.scatter(all_dwd_clean, all_dz_clean, color='gray', s=1, alpha=0.3, label='Relative WD Samples')
        
    # 画拟合线
    plt.plot(y_smooth, z_smooth, 'r-', linewidth=3, label=f'Best Fit ({best_model})')
    
    # 辅助线
    plt.axvline(0, color='k', linestyle=':', alpha=0.5)
    
    plt.title(f'Relative Wind Direction Profile (Veering Analysis)\nWinner: {best_model}', fontsize=14)
    plt.xlabel('Delta Direction (WD - WD_base) [deg]', fontsize=12)
    plt.ylabel('Delta Height (z - z_base) [m]', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 标注公式
    plt.text(0.05, 0.95, f"Universal Formula:\n{formula_str}", transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)
    
    out_img = os.path.join(OUTPUT_DIR, 'WD_Relative_Fit.png')
    plt.savefig(out_img, dpi=300)
    print(f"[图表] 拟合图已保存: {out_img}")

if __name__ == "__main__":
    analyze_wd_relative(FILE_PATH)