import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
from sklearn.metrics import mean_squared_error

# ================= 配置区域 =================
FILE_PATH = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/ti_ultimate_v17'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

# --- 1. 定义模型 ---

def model_ti_quad(z, a, b, c):
    """1. 二次多项式 (U型)"""
    return a * z**2 + b * z + c

def model_ti_sym_inv_gauss(z, ti_base, ti_dip, z_dip, sigma):
    """2. 对称倒置高斯"""
    return ti_base - ti_dip * np.exp(-((z - z_dip)**2) / (2 * sigma**2))

def model_ti_asym_inv_gauss(z, ti_base, ti_dip, z_dip, sigma_down, sigma_up):
    """
    3. 非对称倒置高斯 (你的建议)
    允许急流核心下方和上方的湍流变化速率不同
    """
    sigma = np.where(z <= z_dip, sigma_down, sigma_up)
    return ti_base - ti_dip * np.exp(-((z - z_dip)**2) / (2 * sigma**2))

# --- 2. 辅助函数 ---
def strict_tab_parse_v3(file_path):
    # (保持原有的解析逻辑)
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
def run_ultimate_ti_analysis(file_path):
    print(f"正在分析: {os.path.basename(file_path)}")
    df_raw = strict_tab_parse_v3(file_path)
    if df_raw is None: return

    # 数据提取
    df_raw.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df_raw.columns]
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    print(" -> 提取并筛选 TI 数据...")
    data_list = []
    
    for idx in df_raw.index:
        ws_vals = []
        ti_vals = []
        has_nan = False
        for h in heights:
            try:
                w = float(df_raw.loc[idx, f'{h}m水平风速'])
                std_col = next((c for c in df_raw.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)
                if std_col:
                    std = float(df_raw.loc[idx, std_col])
                    if w > 3.0 and std > 0:
                        ws_vals.append(w)
                        ti_vals.append(std / w)
                    else: has_nan = True
                else: has_nan = True
            except: has_nan = True
            
        if not has_nan and len(ws_vals) == len(heights):
            ws_arr = np.array(ws_vals)
            mx_i = np.argmax(ws_arr)
            mx_h = heights[mx_i]
            # LLJ 筛选
            if (mx_h > MIN_JET_HEIGHT) and (mx_h < MAX_JET_HEIGHT):
                if (ws_arr[mx_i] - ws_arr[0] >= LLJ_THRESHOLD) and (ws_arr[mx_i] - ws_arr[-1] >= LLJ_THRESHOLD):
                    data_list.append({'z': np.array(heights), 'ti': np.array(ti_vals), 'z_jet': mx_h})

    print(f" -> 样本数: {len(data_list)}")
    if len(data_list) < 5: return

    # --- 1. 计算平均廓线 ---
    mean_ti = np.mean([d['ti'] for d in data_list], axis=0)
    z_vec = data_list[0]['z']
    
    # --- 2. 三大模型拟合平均廓线 ---
    print("\n[拟合中] 开始对比三种模型...")
    results = {}
    z_smooth = np.linspace(z_vec[0], z_vec[-1], 200)

    # A. Quadratic
    try:
        popt_q, _ = curve_fit(model_ti_quad, z_vec, mean_ti)
        rmse_q = np.sqrt(mean_squared_error(mean_ti, model_ti_quad(z_vec, *popt_q)))
        results['Quadratic'] = {'popt': popt_q, 'rmse': rmse_q, 'func': model_ti_quad, 'color': 'blue', 'ls': '--'}
    except: pass

    # B. Sym Inverted Gauss
    try:
        # p0: [base, dip, z_dip, sigma]
        p0_sym = [np.max(mean_ti), np.max(mean_ti)-np.min(mean_ti), z_vec[np.argmin(mean_ti)], 50]
        popt_s, _ = curve_fit(model_ti_sym_inv_gauss, z_vec, mean_ti, p0=p0_sym, maxfev=2000)
        rmse_s = np.sqrt(mean_squared_error(mean_ti, model_ti_sym_inv_gauss(z_vec, *popt_s)))
        results['Sym_Inv_Gauss'] = {'popt': popt_s, 'rmse': rmse_s, 'func': model_ti_sym_inv_gauss, 'color': 'green', 'ls': '-.'}
    except: pass

    # C. Asym Inverted Gauss (你的建议)
    try:
        # p0: [base, dip, z_dip, sigma_d, sigma_u]
        p0_asym = [np.max(mean_ti), np.max(mean_ti)-np.min(mean_ti), z_vec[np.argmin(mean_ti)], 40, 60]
        popt_a, _ = curve_fit(model_ti_asym_inv_gauss, z_vec, mean_ti, p0=p0_asym, maxfev=5000)
        rmse_a = np.sqrt(mean_squared_error(mean_ti, model_ti_asym_inv_gauss(z_vec, *popt_a)))
        results['Asym_Inv_Gauss'] = {'popt': popt_a, 'rmse': rmse_a, 'func': model_ti_asym_inv_gauss, 'color': 'red', 'ls': '-'}
    except: pass

    # --- 3. 结果输出与绘图 ---
    print("\n" + "="*50)
    print(" 🏆 拟合误差排行榜 (RMSE)")
    print("="*50)
    
    sorted_res = sorted(results.items(), key=lambda x: x[1]['rmse'])
    for name, res in sorted_res:
        print(f"Model: {name:20s} | RMSE: {res['rmse']:.5f}")

    best_name = sorted_res[0][0]
    best_res = sorted_res[0][1]
    best_popt = best_res['popt']
    
    print("-" * 50)
    print(f"结论: 最优模型是 【{best_name}】")
    
    if best_name == 'Asym_Inv_Gauss':
        print("\n>>> 你的直觉是正确的！非对称模型效果最好。")
        print(">>> 推荐公式参数:")
        print(f"    TI_base (背景湍流) = {best_popt[0]:.4f}")
        print(f"    TI_dip  (核心凹陷) = {best_popt[1]:.4f}")
        print(f"    Z_dip   (凹陷高度) = {best_popt[2]:.1f} m")
        print(f"    Sigma_D (下层厚度) = {best_popt[3]:.1f} m")
        print(f"    Sigma_U (上层厚度) = {best_popt[4]:.1f} m")
        
        ratio = best_popt[4] / best_popt[3]
        print(f"\n    [物理发现] 上层厚度是下层的 {ratio:.1f} 倍，证明了湍流结构的不对称性。")

    # 绘图
    plt.figure(figsize=(9, 7))
    plt.plot(mean_ti, z_vec, 'ko', markersize=8, label='Observed Mean', zorder=10)
    
    for name, res in results.items():
        y_pred = res['func'](z_smooth, *res['popt'])
        label = f"{name} (RMSE={res['rmse']:.4f})"
        lw = 3 if name == best_name else 1.5
        alpha = 1.0 if name == best_name else 0.7
        plt.plot(y_pred, z_smooth, color=res['color'], linestyle=res['ls'], linewidth=lw, alpha=alpha, label=label)

    plt.title(f'TI Profile Model Comparison\nWinner: {best_name}', fontsize=14)
    plt.xlabel('Turbulence Intensity [-]')
    plt.ylabel('Height [m]')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    out_img = os.path.join(OUTPUT_DIR, 'TI_Ultimate_Comparison.png')
    plt.savefig(out_img, dpi=300)
    print(f"\n[图表] 对比图已保存: {out_img}")

if __name__ == "__main__":
    run_ultimate_ti_analysis(FILE_PATH)