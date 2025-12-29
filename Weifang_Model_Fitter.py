import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import re
import os
import glob
import warnings

# ================= 配置区域 =================
# 自动寻找包含"潍坊昌邑"的文件
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
TARGET_STATION = "潍坊昌邑"
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/weifang_case_study'

# 判定标准
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480

# 归一化分箱设置 (用于 WS 和 TI)
BIN_WIDTH = 0.05
MAX_NORM_Z = 2.5
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 1. 定义三大模型 ---

def model_ws_banta_norm(z_norm, alpha, beta):
    """
    归一化 Banta 风速模型
    U/U_jet = (z/Z_jet)^alpha * exp(beta * (1 - z/Z_jet))
    """
    z_norm = np.maximum(z_norm, 1e-6)
    return np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))

def model_ti_banta_inv(z_norm, ti_base, ti_dip, alpha, beta):
    """
    倒置 Banta 湍流模型 (输入为归一化高度)
    TI = Base - Dip * Shape
    """
    z_norm = np.maximum(z_norm, 1e-6)
    shape = np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))
    return ti_base - ti_dip * shape

def model_wd_linear(z, wd_base, k):
    """
    线性风向模型 (输入为物理高度)
    WD = Base + k * (z - z_base)
    注意：为了拟合方便，我们在函数外处理 z - z_base
    """
    return wd_base + k * z

# --- 2. 辅助工具 ---

def unwrap_deg(degrees):
    """解缠绕"""
    rads = np.radians(degrees)
    unwrapped = np.unwrap(rads)
    return np.degrees(unwrapped)

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

# --- 3. 核心处理逻辑 ---

def process_station():
    # 1. 寻找文件
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    target_file = next((f for f in files if TARGET_STATION in os.path.basename(f)), None)
    
    if not target_file:
        print(f"Error: 未找到包含 '{TARGET_STATION}' 的数据文件！")
        return

    print(f"正在分析目标场站: {os.path.basename(target_file)}")
    df = strict_tab_parse(target_file)
    if df is None: return

    # 清洗列名
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    
    # 提取高度层
    speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    print(f"  -> 识别高度层: {heights}")

    # 准备容器
    norm_z_list = []   # 存放 z / Z_jet
    norm_ws_list = []  # 存放 U / U_jet
    norm_ti_list = []  # 存放 TI
    
    phys_z_wd_list = [] # 存放 z (物理高度)
    delta_wd_list = []  # 存放 WD - WD_base
    
    stats_zjet = []
    stats_ujet = []

    # 逐行处理
    print("  -> 正在提取并归一化数据...")
    for idx in df.index:
        # 提取当前时刻的廓线
        u_profile = []
        wd_profile = []
        ti_profile = []
        
        valid_h = []
        
        for h in heights:
            try:
                # 风速
                ws_col = [c for c in df.columns if f'{h}m水平风速' in c and '最大' not in c][0]
                u = float(df.loc[idx, ws_col])
                
                # 风向
                wd_col = next((c for c in df.columns if str(h) in c and ('风向' in c or 'Direction' in c) and '最大' not in c), None)
                d = float(df.loc[idx, wd_col]) if wd_col else np.nan
                
                # TI
                std_col = next((c for c in df.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)
                std = float(df.loc[idx, std_col]) if std_col else np.nan
                
                if u > 3.0: # 基础过滤
                    u_profile.append(u)
                    wd_profile.append(d)
                    ti_profile.append(std/u if std>0 else np.nan)
                    valid_h.append(h)
            except: pass
            
        if not u_profile: continue
        
        u_arr = np.array(u_profile)
        h_arr = np.array(valid_h)
        
        # 判定 LLJ
        mx_i = np.argmax(u_arr)
        z_jet = h_arr[mx_i]
        u_jet = u_arr[mx_i]
        
        if (z_jet > MIN_JET_HEIGHT) and (z_jet < MAX_JET_HEIGHT):
            if (u_jet - u_arr[0] >= LLJ_THRESHOLD) and (u_jet - u_arr[-1] >= LLJ_THRESHOLD):
                # === 收集数据 ===
                stats_zjet.append(z_jet)
                stats_ujet.append(u_jet)
                
                # 1. WS & TI (归一化)
                z_norm = h_arr / z_jet
                u_norm = u_arr / u_jet
                
                norm_z_list.extend(z_norm)
                norm_ws_list.extend(u_norm)
                norm_ti_list.extend(ti_profile)
                
                # 2. WD (物理高度，计算相对偏转)
                wd_arr = np.array(wd_profile)
                if not np.isnan(wd_arr).any():
                    wd_cont = unwrap_deg(wd_arr)
                    wd_base = wd_cont[0] # 以最底层为基准
                    
                    # 记录物理高度的偏转量
                    phys_z_wd_list.extend(h_arr - h_arr[0]) # 相对高度差
                    delta_wd_list.extend(wd_cont - wd_base)

    if not norm_z_list:
        print("未找到有效急流样本")
        return

    # --- 数据分箱与平均 (Binning) ---
    print("  -> 执行分箱统计与拟合...")
    
    # 1. WS & TI Binning
    norm_z_arr = np.array(norm_z_list)
    norm_ws_arr = np.array(norm_ws_list)
    norm_ti_arr = np.array(norm_ti_list)
    
    # 过滤无效 TI
    mask_ti = ~np.isnan(norm_ti_arr)
    
    bins = np.arange(0, MAX_NORM_Z, BIN_WIDTH)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    binned_ws = []
    binned_ti = []
    valid_bins_ws = []
    valid_bins_ti = []
    
    for i in range(len(bins)-1):
        # WS
        mask = (norm_z_arr >= bins[i]) & (norm_z_arr < bins[i+1])
        if np.sum(mask) > 10:
            binned_ws.append(np.mean(norm_ws_arr[mask]))
            valid_bins_ws.append(bin_centers[i])
            
        # TI
        mask_t = mask & mask_ti
        if np.sum(mask_t) > 10:
            binned_ti.append(np.mean(norm_ti_arr[mask_t]))
            valid_bins_ti.append(bin_centers[i])
            
    # 2. WD Averaging (Physical Height Bins)
    phys_z_arr = np.array(phys_z_wd_list)
    delta_wd_arr = np.array(delta_wd_list)
    
    unique_h = sorted(list(set(phys_z_arr)))
    mean_delta_wd = []
    valid_h_wd = []
    
    for h in unique_h:
        mask = (phys_z_arr == h)
        if np.sum(mask) > 5:
            mean_delta_wd.append(np.mean(delta_wd_arr[mask]))
            valid_h_wd.append(h)

    # --- 拟合 ---
    results_txt = []
    results_txt.append(f"=== {TARGET_STATION} 风参模型拟合报告 ===")
    results_txt.append(f"样本数: {len(stats_zjet)}")
    results_txt.append(f"平均急流高度 Z_jet: {np.mean(stats_zjet):.1f} m")
    results_txt.append(f"平均急流风速 U_jet: {np.mean(stats_ujet):.1f} m/s")
    results_txt.append("-" * 30)

    # 1. Fit WS (Banta)
    popt_ws, _ = curve_fit(model_ws_banta_norm, valid_bins_ws, binned_ws, p0=[1.0, 1.0])
    results_txt.append("[1. 风速模型 - Banta]")
    results_txt.append(f"  公式: U/U_jet = (z/Z_jet)^alpha * exp(beta*(1 - z/Z_jet))")
    results_txt.append(f"  Alpha (切变指数): {popt_ws[0]:.4f}")
    results_txt.append(f"  Beta  (衰减指数): {popt_ws[1]:.4f}")
    
    # 2. Fit TI (Inv Banta)
    # p0: base, dip, alpha, beta
    p0_ti = [np.max(binned_ti), 0.05, 1.0, 1.0] 
    try:
        popt_ti, _ = curve_fit(model_ti_banta_inv, valid_bins_ti, binned_ti, p0=p0_ti, maxfev=5000)
    except:
        popt_ti = [np.nan]*4
        
    results_txt.append("\n[2. 湍流模型 - Inverted Banta]")
    results_txt.append(f"  公式: TI = Base - Dip * (z/Z_jet)^alpha * exp(beta*(1 - z/Z_jet))")
    results_txt.append(f"  TI_base (基底湍流): {popt_ti[0]:.4f}")
    results_txt.append(f"  Delta_TI (凹陷深度): {popt_ti[1]:.4f}")
    results_txt.append(f"  Alpha (形状参数): {popt_ti[2]:.4f}")
    results_txt.append(f"  Beta  (形状参数): {popt_ti[3]:.4f}")

    # 3. Fit WD (Linear)
    # Fit y = k * x (intercept is 0 since we use delta)
    def linear_origin(x, k): return k * x
    popt_wd, _ = curve_fit(linear_origin, valid_h_wd, mean_delta_wd)
    
    results_txt.append("\n[3. 风向模型 - Linear]")
    results_txt.append(f"  公式: WD(z) = WD_base + k * (z - z_base)")
    results_txt.append(f"  k (偏转率): {popt_wd[0]:.4f} deg/m")
    results_txt.append(f"  说明: 每上升100m，风向偏转约 {popt_wd[0]*100:.1f} 度")

    # --- 打印报告 ---
    print("\n".join(results_txt))
    
    # --- 绘图 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: WS
    z_smooth = np.linspace(0, 2.5, 100)
    axes[0].scatter(binned_ws, valid_bins_ws, c='k', label='Observed (Norm)')
    axes[0].plot(model_ws_banta_norm(z_smooth, *popt_ws), z_smooth, 'r-', lw=3, label='Banta Fit')
    axes[0].set_title(f'Wind Speed (Normalized)\nalpha={popt_ws[0]:.2f}, beta={popt_ws[1]:.2f}')
    axes[0].set_xlabel('U / U_jet')
    axes[0].set_ylabel('z / Z_jet')
    axes[0].axhline(1, ls=':', c='gray')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: TI
    axes[1].scatter(binned_ti, valid_bins_ti, c='k', label='Observed (Norm)')
    if not np.isnan(popt_ti[0]):
        axes[1].plot(model_ti_banta_inv(z_smooth, *popt_ti), z_smooth, 'b-', lw=3, label='Inv-Banta Fit')
    axes[1].set_title(f'Turbulence Intensity (Normalized)\nDip={popt_ti[1]:.3f}')
    axes[1].set_xlabel('TI [-]')
    axes[1].set_ylabel('z / Z_jet')
    axes[1].axhline(1, ls=':', c='gray')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot 3: WD
    axes[2].scatter(mean_delta_wd, valid_h_wd, c='k', label='Observed (Mean)')
    z_phys_smooth = np.linspace(0, max(valid_h_wd), 100)
    axes[2].plot(popt_wd[0]*z_phys_smooth, z_phys_smooth, 'g-', lw=3, label='Linear Fit')
    axes[2].set_title(f'Wind Direction Veering\nRate={popt_wd[0]:.4f} deg/m')
    axes[2].set_xlabel('Delta WD [deg]')
    axes[2].set_ylabel('Delta Height [m]')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Weifang_Model_Fits.png'), dpi=150)
    print(f"\n图表已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_station()