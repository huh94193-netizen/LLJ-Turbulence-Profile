import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
import glob
import warnings
import time

# ================= 配置区域 =================
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/all_stations_params_final'

# 判定标准
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 60
MAX_JET_HEIGHT = 480

# 绘图设置
PLOT_FITS = True  # 是否保存每个场站的拟合图
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
if PLOT_FITS:
    os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
warnings.filterwarnings('ignore')

# --- 1. 模型定义 ---
def model_ws_banta_norm(z_norm, alpha, beta):
    """Banta 风速模型"""
    z_norm = np.maximum(z_norm, 1e-6)
    return np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))

def model_ti_banta_inv(z_norm, ti_base, ti_dip, alpha, beta):
    """倒置 Banta 湍流模型"""
    z_norm = np.maximum(z_norm, 1e-6)
    shape = np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))
    return ti_base - ti_dip * shape

def unwrap_deg(degrees):
    """风向解缠绕"""
    rads = np.radians(degrees)
    unwrapped = np.unwrap(rads)
    return np.degrees(unwrapped)

# --- 2. 辅助工具 ---
def is_clean_col(col_name):
    """黑名单过滤器：排除最大、最小、标准差等干扰列"""
    blacklist = ['最大', '最小', '偏差', '矢量', '标准差', 'Max', 'Min', 'Std', 'Dev', 'Vector', 'Gust', 'Sigma']
    for bad in blacklist:
        if bad.lower() in col_name.lower(): return False
    return True

def get_pure_col(columns, height, keyword_list):
    """查找最短、最纯净的列名"""
    candidates = []
    for c in columns:
        if f'{height}m' not in c: continue
        has_keyword = any(k in c for k in keyword_list)
        if not has_keyword: continue
        if is_clean_col(c): candidates.append(c)
    if not candidates: return None
    candidates.sort(key=len)
    return candidates[0]

def strict_tab_parse(file_path):
    """健壮的文件读取"""
    encodings = ['utf-8', 'gbk', 'gb18030', 'latin-1']
    for enc in encodings:
        try:
            temp = pd.read_csv(file_path, sep='\t', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp.columns): return temp
            temp = pd.read_csv(file_path, sep='\s+', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp.columns): return temp
        except: continue
    return None

# --- 3. 单场站处理函数 (核心逻辑) ---
def analyze_station(file_path):
    station_name = os.path.basename(file_path).split('-')[0] # 假设文件名格式 "站名-xxxx.txt"
    print(f"[{station_name}] 正在处理...", end=' ')
    
    start_time = time.time()
    df = strict_tab_parse(file_path)
    if df is None: 
        print("读取失败")
        return None
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    # --- 识别高度 ---
    clean_ws_cols_temp = [c for c in df.columns if ('水平风速' in c or 'Speed' in c) and is_clean_col(c)]
    raw_heights = []
    for c in clean_ws_cols_temp:
        m = re.search(r'(\d+)m', c)
        if m: raw_heights.append(int(m.group(1)))
    heights = sorted(list(set(raw_heights)))
    
    if len(heights) < 5:
        print("有效高度层不足")
        return None

    # --- 映射列名 ---
    map_ws, map_wd, map_std = [], [], []
    valid_indices = []
    for i, h in enumerate(heights):
        c_ws = get_pure_col(df.columns, h, ['水平风速', 'Speed'])
        c_wd = get_pure_col(df.columns, h, ['风向', 'Direction'])
        c_std = next((c for c in df.columns if f'{h}m' in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)
        if c_ws:
            map_ws.append(c_ws); map_wd.append(c_wd); map_std.append(c_std)
            valid_indices.append(i)
        else:
            map_ws.append(None); map_wd.append(None); map_std.append(None)
            
    final_ws_cols = [c for c in map_ws if c is not None]
    final_heights = np.array([heights[i] for i in range(len(heights)) if map_ws[i] is not None])

    # --- 矩阵计算 ---
    mat_ws = df[final_ws_cols].apply(pd.to_numeric, errors='coerce').values
    
    mat_wd = np.full(mat_ws.shape, np.nan)
    for i, idx in enumerate(valid_indices):
        col = map_wd[idx]
        if col: mat_wd[:, i] = pd.to_numeric(df[col], errors='coerce').values
            
    mat_std = np.full(mat_ws.shape, np.nan)
    for i, idx in enumerate(valid_indices):
        col = map_std[idx]
        if col: mat_std[:, i] = pd.to_numeric(df[col], errors='coerce').values

    with np.errstate(divide='ignore', invalid='ignore'):
        mat_ti = mat_std / mat_ws
        mat_ti[mat_ws < 3.0] = np.nan 
        mat_ti[mat_ti > 1.0] = np.nan # 过滤 TI > 100% 的异常点

    # --- 向量化筛选 ---
    nan_mask = np.isnan(mat_ws)
    temp_ws = mat_ws.copy()
    temp_ws[nan_mask] = -999.0
    
    max_ws_indices = np.argmax(temp_ws, axis=1)
    max_ws_values = np.max(temp_ws, axis=1)
    z_jet_arr = final_heights[max_ws_indices]
    
    # 动态顶层与底层识别
    valid_mask = ~nan_mask
    if not valid_mask.any(): return None
    
    first_valid_idx = np.argmax(valid_mask, axis=1) 
    last_valid_idx = mat_ws.shape[1] - 1 - np.argmax(valid_mask[:, ::-1], axis=1)
    
    row_indices = np.arange(mat_ws.shape[0])
    ws_bottom = mat_ws[row_indices, first_valid_idx]
    ws_top = mat_ws[row_indices, last_valid_idx]
    
    mask_valid_jet = (
        ~(nan_mask.all(axis=1)) &
        (z_jet_arr > MIN_JET_HEIGHT) & 
        (z_jet_arr < MAX_JET_HEIGHT) &
        ((max_ws_values - ws_bottom) >= LLJ_THRESHOLD) &
        ((max_ws_values - ws_top) >= LLJ_THRESHOLD) &
        (max_ws_indices != first_valid_idx) &
        (max_ws_indices != last_valid_idx)
    )
    
    valid_rows = np.where(mask_valid_jet)[0]
    n_samples = len(valid_rows)
    if n_samples < 50: 
        print(f"有效样本过少 ({n_samples})")
        return None

    # --- 归一化 ---
    sel_ws = mat_ws[valid_rows, :]
    sel_ti = mat_ti[valid_rows, :]
    sel_wd = mat_wd[valid_rows, :]
    sel_z_jet = z_jet_arr[valid_rows].reshape(-1, 1)
    sel_u_jet = max_ws_values[valid_rows].reshape(-1, 1)
    
    norm_z_matrix = final_heights.reshape(1, -1) / sel_z_jet
    norm_ws_matrix = sel_ws / sel_u_jet
    
    flat_norm_z = norm_z_matrix.flatten()
    flat_norm_ws = norm_ws_matrix.flatten()
    flat_norm_ti = sel_ti.flatten()
    
    # 风向解缠绕 (WD Unwrap)
    phys_z_wd_list = []
    delta_wd_list = []
    for i in range(len(valid_rows)):
        row_wd = sel_wd[i, :]
        mask_v = ~np.isnan(row_wd)
        if np.sum(mask_v) > 2:
            wd_valid = row_wd[mask_v]
            h_valid = final_heights[mask_v]
            wd_unwrapped = unwrap_deg(wd_valid)
            phys_z_wd_list.extend(h_valid - h_valid[0])
            delta_wd_list.extend(wd_unwrapped - wd_unwrapped[0])

    # --- 拟合准备 (Binning) ---
    # Binning WS
    mask_ws = ~np.isnan(flat_norm_ws)
    flat_norm_z_ws = flat_norm_z[mask_ws]
    flat_norm_ws = flat_norm_ws[mask_ws]
    
    bins = np.arange(0, 2.5, 0.05)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    inds = np.digitize(flat_norm_z_ws, bins)
    binned_ws = [np.mean(flat_norm_ws[inds == i]) for i in range(1, len(bins)) if len(flat_norm_ws[inds == i]) > 10]
    valid_bins_ws = [bin_centers[i-1] for i in range(1, len(bins)) if len(flat_norm_ws[inds == i]) > 10]
            
    # Binning TI
    mask_ti = ~np.isnan(flat_norm_ti)
    flat_norm_z_ti = flat_norm_z[mask_ti]
    flat_norm_ti = flat_norm_ti[mask_ti]
    inds_ti = np.digitize(flat_norm_z_ti, bins)
    binned_ti = [np.mean(flat_norm_ti[inds_ti == i]) for i in range(1, len(bins)) if len(flat_norm_ti[inds_ti == i]) > 10]
    valid_bins_ti = [bin_centers[i-1] for i in range(1, len(bins)) if len(flat_norm_ti[inds_ti == i]) > 10]

    # Binning WD
    phys_z_wd_arr = np.array(phys_z_wd_list)
    delta_wd_arr = np.array(delta_wd_list)
    unique_h = sorted(list(set(phys_z_wd_arr)))
    mean_delta_wd = []
    valid_h_wd = []
    for h in unique_h:
        mask = (phys_z_wd_arr == h)
        if np.sum(mask) > 5:
            mean_delta_wd.append(np.mean(delta_wd_arr[mask]))
            valid_h_wd.append(h)

    # --- 核心拟合 ---
    result = {
        'Station': station_name,
        'N_Samples': n_samples,
        'Z_jet_avg': np.mean(sel_z_jet),
        'U_jet_avg': np.mean(sel_u_jet)
    }

    # 1. WS Fit
    try:
        popt_ws, _ = curve_fit(model_ws_banta_norm, valid_bins_ws, binned_ws, p0=[1.0, 1.0])
        result['WS_Alpha'] = popt_ws[0]
        result['WS_Beta'] = popt_ws[1]
    except: 
        result['WS_Alpha'] = np.nan; result['WS_Beta'] = np.nan
        popt_ws = [np.nan, np.nan]

    # 2. TI Fit (关键修复：Bounds)
    try:
        # p0 = [Base, Dip, Alpha, Beta]
        p0_ti = [0.15, 0.05, 1.0, 1.0]
        # 【物理锁】Base 上限设为 0.4，防止飞到 1.0
        bounds_ti = ([0, 0, 0, 0], [0.4, 1, 10, 10]) 
        
        popt_ti, _ = curve_fit(model_ti_banta_inv, valid_bins_ti, binned_ti, p0=p0_ti, bounds=bounds_ti, maxfev=5000)
        result['TI_Base'] = popt_ti[0]
        result['TI_Dip'] = popt_ti[1]
        result['TI_Alpha'] = popt_ti[2]
        result['TI_Beta'] = popt_ti[3]
    except:
        result['TI_Base'] = np.nan; result['TI_Dip'] = np.nan
        result['TI_Alpha'] = np.nan; result['TI_Beta'] = np.nan
        popt_ti = [np.nan]*4

    # 3. WD Fit
    try:
        def linear_origin(x, k): return k * x
        popt_wd, _ = curve_fit(linear_origin, valid_h_wd, mean_delta_wd)
        result['WD_Linear_k'] = popt_wd[0]
    except:
        result['WD_Linear_k'] = np.nan
        popt_wd = [np.nan]

    print(f"完成 (耗时 {time.time()-start_time:.1f}s)")
    
    # --- 绘图 (可选) ---
    if PLOT_FITS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        z_smooth = np.linspace(0, 2.5, 100)
        
        # WS Plot
        axes[0].scatter(binned_ws, valid_bins_ws, c='k', s=15)
        if not np.isnan(popt_ws[0]):
            axes[0].plot(model_ws_banta_norm(z_smooth, *popt_ws), z_smooth, 'r-', lw=2)
        axes[0].set_title(f'{station_name} WS (Banta)\nα={result["WS_Alpha"]:.2f}, β={result["WS_Beta"]:.2f}')
        axes[0].set_xlabel('U/U_jet')
        axes[0].set_ylabel('z/Z_jet')
        axes[0].axhline(1, ls=':', c='gray')
        axes[0].grid(alpha=0.3)
        
        # TI Plot
        axes[1].scatter(binned_ti, valid_bins_ti, c='k', s=15)
        if not np.isnan(popt_ti[0]):
            axes[1].plot(model_ti_banta_inv(z_smooth, *popt_ti), z_smooth, 'b-', lw=2)
        axes[1].set_title(f'{station_name} TI (Inv-Banta)\nDip={result["TI_Dip"]:.3f}, Base={result["TI_Base"]:.2f}')
        axes[1].set_xlabel('TI [-]')
        axes[1].axhline(1, ls=':', c='gray')
        axes[1].grid(alpha=0.3)
        
        # WD Plot
        axes[2].scatter(mean_delta_wd, valid_h_wd, c='k', s=15)
        if not np.isnan(popt_wd[0]):
            zp = np.linspace(0, max(valid_h_wd) if valid_h_wd else 100, 100)
            axes[2].plot(popt_wd[0]*zp, zp, 'g-', lw=2)
        axes[2].set_title(f'{station_name} WD (Linear)\nk={result["WD_Linear_k"]:.4f}')
        axes[2].set_xlabel('Delta WD [deg]')
        axes[2].set_ylabel('Delta Height [m]')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'plots', f'{station_name}_Fits.png'), dpi=100)
        plt.close()

    return result

# --- 4. 主程序 ---
def main():
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    print(f"找到 {len(files)} 个数据文件，开始批量处理...")
    
    all_results = []
    
    for f in files:
        try:
            res = analyze_station(f)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"\n[Error] 处理 {os.path.basename(f)} 时出错: {e}")
            continue
            
    if not all_results:
        print("未生成任何结果。")
        return
        
    # --- 汇总输出 ---
    df_res = pd.DataFrame(all_results)
    
    # 调整列顺序
    cols = ['Station', 'N_Samples', 'Z_jet_avg', 'U_jet_avg', 
            'WS_Alpha', 'WS_Beta', 
            'TI_Base', 'TI_Dip', 'TI_Alpha', 'TI_Beta', 
            'WD_Linear_k']
    
    # 确保所有列都存在
    for c in cols:
        if c not in df_res.columns: df_res[c] = np.nan
            
    df_res = df_res[cols]
    
    out_file = os.path.join(OUTPUT_DIR, 'All_Stations_Parameters_Final.xlsx')
    df_res.to_excel(out_file, index=False)
    
    print("="*50)
    print("全场站处理完成！")
    print(f"结果表已保存至: {out_file}")
    print(f"拟合图已保存至: {os.path.join(OUTPUT_DIR, 'plots')}")

if __name__ == "__main__":
    main()