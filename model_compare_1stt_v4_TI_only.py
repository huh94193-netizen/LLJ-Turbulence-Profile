import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import re
import os
from sklearn.metrics import mean_squared_error

# ================= 配置区域 =================
# 替换你的数据文件路径
FILE_PATH = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/ti_shape_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 质量控制
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

# --- 1. 定义湍流模型候选者 ---

def model_ti_power(z, a, b):
    """1. 传统幂律 (单调衰减)"""
    return a * np.power(z, -b)

def model_ti_quadratic(z, a, b, c):
    """2. 二次多项式 (U型/抛物线)"""
    return a * z**2 + b * z + c

def model_ti_inverted_gauss(z, ti_base, ti_dip, z_dip, sigma):
    """3. 倒置高斯 (深井模型) - 模拟急流核心处的湍流凹陷"""
    # ti_base: 背景湍流(高值)
    # ti_dip: 凹陷深度
    # z_dip: 凹陷中心高度
    return ti_base - ti_dip * np.exp(-((z - z_dip)**2) / (2 * sigma**2))

# --- 2. 数据处理工具 ---
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

# --- 3. 核心逻辑 ---
def analyze_ti_shape(file_path):
    print(f"正在分析文件: {os.path.basename(file_path)}")
    df_raw = strict_tab_parse_v3(file_path)
    if df_raw is None: return

    # 提取列
    df_raw.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df_raw.columns]
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    # 提取并计算 TI
    print(" -> 正在提取数据并计算 TI...")
    data_list = []
    
    for idx in df_raw.index:
        ws_vals = []
        ti_vals = []
        has_nan = False
        
        for h in heights:
            try:
                w = float(df_raw.loc[idx, f'{h}m水平风速'])
                # 找对应的 Std
                std_col = next((c for c in df_raw.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)
                if std_col:
                    std = float(df_raw.loc[idx, std_col])
                    if w > 3.0 and std > 0: # 过滤低风速和无效值
                        ws_vals.append(w)
                        ti_vals.append(std / w)
                    else: has_nan = True
                else: has_nan = True
            except: has_nan = True
            
        if not has_nan and len(ws_vals) == len(heights):
            ws_arr = np.array(ws_vals)
            # LLJ 判定
            mx_i = np.argmax(ws_arr)
            mx_h = heights[mx_i]
            if (mx_h > MIN_JET_HEIGHT) and (mx_h < MAX_JET_HEIGHT):
                if (ws_arr[mx_i] - ws_arr[0] >= LLJ_THRESHOLD) and (ws_arr[mx_i] - ws_arr[-1] >= LLJ_THRESHOLD):
                    data_list.append({
                        'z': np.array(heights),
                        'ti': np.array(ti_vals),
                        'z_jet': mx_h
                    })

    print(f" -> 捕获 LLJ 样本: {len(data_list)} 个")
    if len(data_list) < 10: return

    # --- Step 1: 竞技场 (Per Event Comparison) ---
    print("\n[阶段 1] 模型竞技场 (Per-Event Fitting)...")
    errors = {'PowerLaw': [], 'Quadratic': [], 'InvertedGauss': []}
    
    for ev in data_list:
        z = ev['z']
        ti = ev['ti']
        
        # 1. Power Law
        try:
            popt, _ = curve_fit(model_ti_power, z, ti, p0=[0.1, 0.1], maxfev=800)
            rmse = np.sqrt(mean_squared_error(ti, model_ti_power(z, *popt)))
            errors['PowerLaw'].append(rmse)
        except: pass
        
        # 2. Quadratic
        try:
            popt, _ = curve_fit(model_ti_quadratic, z, ti, maxfev=800)
            rmse = np.sqrt(mean_squared_error(ti, model_ti_quadratic(z, *popt)))
            errors['Quadratic'].append(rmse)
        except: pass
        
        # 3. Inverted Gauss
        try:
            # p0: [base, dip_depth, z_dip, sigma]
            p0 = [np.max(ti), np.max(ti)-np.min(ti), ev['z_jet'], 50]
            popt, _ = curve_fit(model_ti_inverted_gauss, z, ti, p0=p0, maxfev=1000)
            rmse = np.sqrt(mean_squared_error(ti, model_ti_inverted_gauss(z, *popt)))
            errors['InvertedGauss'].append(rmse)
        except: pass

    # 打印竞技结果
    print("-" * 40)
    best_model_name = "PowerLaw"
    min_median_error = 999
    
    for name, errs in errors.items():
        if not errs: continue
        med_rmse = np.median(errs)
        print(f"Model: {name:15s} | Median RMSE: {med_rmse:.5f}")
        if med_rmse < min_median_error:
            min_median_error = med_rmse
            best_model_name = name
    print("-" * 40)
    print(f"🏆 冠军模型: {best_model_name}")

    # --- Step 2: 全局拟合 (Global Fit on Mean Profile) ---
    print("\n[阶段 2] 生成该场站的 TI 通用公式...")
    
    # 计算平均 TI 廓线
    mean_ti = np.mean([d['ti'] for d in data_list], axis=0)
    z_vec = data_list[0]['z']
    
    # 拟合 Quadratic (最稳健的 U 型描述)
    popt_quad, _ = curve_fit(model_ti_quadratic, z_vec, mean_ti)
    
    # 拟合 Inverted Gauss (物理意义更好)
    try:
        p0_ig = [np.max(mean_ti), np.max(mean_ti)-np.min(mean_ti), z_vec[np.argmin(mean_ti)], 50]
        popt_ig, _ = curve_fit(model_ti_inverted_gauss, z_vec, mean_ti, p0=p0_ig, maxfev=5000)
    except:
        popt_ig = None

    # --- Step 3: 绘图与输出 ---
    plt.figure(figsize=(10, 8))
    
    # 画原始平均点
    plt.plot(mean_ti, z_vec, 'ko', markersize=8, label='Observed Mean TI')
    
    # 画 Quadratic 曲线
    z_smooth = np.linspace(z_vec[0], z_vec[-1], 200)
    ti_quad_pred = model_ti_quadratic(z_smooth, *popt_quad)
    plt.plot(ti_quad_pred, z_smooth, 'b--', linewidth=2, label='Quadratic Fit (U-Shape)')
    
    # 画 Inverted Gauss 曲线
    if popt_ig is not None:
        ti_ig_pred = model_ti_inverted_gauss(z_smooth, *popt_ig)
        plt.plot(ti_ig_pred, z_smooth, 'r-', linewidth=2, label='Inverted Gaussian (Dip Model)')
        
    plt.title(f'TI Profile Shape Analysis - {best_model_name} Wins', fontsize=14)
    plt.xlabel('Turbulence Intensity [-]', fontsize=12)
    plt.ylabel('Height [m]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    out_img = os.path.join(OUTPUT_DIR, 'TI_Shape_Fit.png')
    plt.savefig(out_img, dpi=300)
    print(f"[图表] 拟合对比图已保存: {out_img}")
    
    # 输出公式
    print("\n" + "="*60)
    print(" >>> 推荐使用的 TI 拟合公式 <<<")
    print("="*60)
    
    if popt_ig is not None:
        print(f"【首选：倒置高斯模型 (物理意义最佳)】")
        print(f"说明: 描述了背景湍流中，因急流核心稳定而产生的'凹陷'。")
        print(f"Formula: TI(z) = {popt_ig[0]:.4f} - {popt_ig[1]:.4f} * exp( -((z - {popt_ig[2]:.1f})^2) / (2 * {popt_ig[3]:.1f}^2) )")
        print(f"    - Base TI (基准湍流): {popt_ig[0]:.4f}")
        print(f"    - Dip Depth (凹陷深度): {popt_ig[1]:.4f}")
        print(f"    - Dip Height (凹陷高度): {popt_ig[2]:.1f} m (通常对应急流核心)")
        print("-" * 30)

    print(f"【备选：二次多项式 (简单稳健)】")
    print(f"Formula: TI(z) = {popt_quad[0]:.2e} * z^2 + {popt_quad[1]:.2e} * z + {popt_quad[2]:.4f}")

if __name__ == "__main__":
    analyze_ti_shape(FILE_PATH)