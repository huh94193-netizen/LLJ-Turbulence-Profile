# -*- coding: utf-8 -*-
"""
Downburst Detector V4 (Final Fix based on test_read.py)
修复：严格复刻 test_read.py 的读取逻辑，解决编码和分隔符报错
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
import logging
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

warnings.filterwarnings('ignore')

# ================= 1. 全局配置 =================
BASE_DIR = '/home/huxun/02_LLJ'
DATA_PATH = os.path.join(BASE_DIR, 'exported_data')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
PLOT_DIR = os.path.join(RESULT_DIR, 'plots_WK_verified')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

for d in [PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

CSV_OUTPUT = os.path.join(RESULT_DIR, 'downburst_final_v4.csv')
LOG_FILE = os.path.join(LOG_DIR, 'process_v4.log')

# --- 筛选阈值 ---
CONFIG = {
    'min_speed': 12.0,           # [m/s] 初筛最小风速
    'max_nose_height': 200,      # [m] 鼻尖高度上限
    'min_turbulence': 0.10,      # [TI] 湍流强度
    'temp_drop_threshold': -1.0, # [℃] 降温阈值
    
    # Wood-Kwok 模型参数 (物理校验)
    'wk_max_delta': 800.0,       # [m] 射流厚度上限 (剔除LLJ)
    'wk_min_r2': 0.60,           # [0-1] 最小拟合优度
}

# ================= 2. Wood-Kwok 物理模型 =================
def model_wood_kwok(z, u_max, delta):
    z = np.maximum(z, 0.1)
    term1 = 1.55 * u_max
    term2 = np.power(z / delta, 1.0/6.0)
    term3 = 1.0 - erf(0.7 * z / delta)
    return term1 * term2 * term3

def verify_candidate_with_physics(heights, u_values):
    """Wood-Kwok 曲线拟合校验"""
    mask = ~np.isnan(u_values)
    z_valid = np.array(heights)[mask]
    u_valid = u_values[mask]
    
    if len(u_valid) < 4: return False, {}, 0.0

    u_max_obs = np.max(u_valid)
    
    try:
        popt, _ = curve_fit(model_wood_kwok, z_valid, u_valid, 
                               p0=[u_max_obs, 300], 
                               bounds=([5, 50], [100, 2500]))
        
        u_max_fit, delta_fit = popt
        
        # 计算 R2
        u_pred = model_wood_kwok(z_valid, *popt)
        ss_res = np.sum((u_valid - u_pred) ** 2)
        ss_tot = np.sum((u_valid - np.mean(u_valid)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # 物理判据
        if r2 < CONFIG['wk_min_r2']: return False, {'r2': r2}, r2
        if delta_fit > CONFIG['wk_max_delta']: return False, {'delta': delta_fit}, r2
        
        return True, {'u_max_fit': u_max_fit, 'delta': delta_fit, 'r2': r2}, r2
        
    except Exception:
        return False, {}, 0.0

# ================= 3. 绘图与日志 =================
logger = logging.getLogger('DB_V4')
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
logger.addHandler(logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'))
logger.addHandler(logging.StreamHandler())

def plot_verification(station, timestamp, heights, u_obs, wk_params, info):
    try:
        z_smooth = np.linspace(0, heights[-1], 100)
        u_smooth = model_wood_kwok(z_smooth, wk_params['u_max_fit'], wk_params['delta'])
        
        plt.figure(figsize=(6, 8))
        plt.plot(u_obs, heights, 'ko', markersize=8, label='Obs')
        plt.plot(u_smooth, z_smooth, 'r-', linewidth=2.5, alpha=0.8,
                 label=f"Fit R2={wk_params['r2']:.2f}\n$\delta$={wk_params['delta']:.0f}m")
        
        title = (f"{station} | {timestamp}\n"
                 f"TempDrop: {info.get('temp_drop',0):.1f}C | Umax: {info.get('u_max',0):.1f}")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        
        safe_time = str(timestamp).replace(':','').replace(' ','_')
        plt.savefig(os.path.join(PLOT_DIR, f"{station}_{safe_time}_WK.png"), dpi=100)
        plt.close()
    except Exception as e:
        logger.error(f"Plot error: {e}")

# ================= 4. 核心处理 (严格复刻 test_read.py) =================
def process_file(filepath):
    station = os.path.basename(filepath).split('-')[0]
    logger.info(f"Scanning: {station} ...")
    
    try:
        # === 关键修改：严格按照 test_read.py 的方式读取 ===
        # 不指定 encoding，默认 utf-8；使用 python 引擎处理 \s+
        df = pd.read_csv(
            filepath,
            skiprows=12,    # 跳过前12行注释
            sep=r'\s+',     # 空格分隔
            engine='python',
            on_bad_lines='skip'
        )
        
        # 清洗列名 (去空格和引号)
        df.columns = [str(c).strip().replace('"', '') for c in df.columns]
        
        all_cols = df.columns.tolist()
        
        # 识别风速列 (Gust > Speed)
        has_gust = any(('最大' in c or 'Gust' in c) for c in all_cols if ('风速' in c or 'Speed' in c))
        keyword = '最大' if has_gust else '水平'
        
        # 提取高度层
        # 逻辑：列名包含 keyword 且包含 'm' 且不包含 '方向'
        ws_cols = [c for c in all_cols if keyword in c and 'm' in c and '方向' not in c and ('风速' in c or 'Speed' in c)]
        
        if not ws_cols:
            # 降级尝试
            ws_cols = [c for c in all_cols if 'm' in c and '方向' not in c and ('风速' in c or 'Speed' in c)]
        
        if not ws_cols:
            logger.warning("  -> No wind speed columns found.")
            return []
            
        heights = sorted(list(set([int(c.split('m')[0]) for c in ws_cols if c.split('m')[0].isdigit()])))
        
        if len(heights) < 4: return []

        # 时间处理
        # 你的 test_read.py 显示第一列通常是 Date/Time，但也可能是其他名字
        # 这里尝试自动找时间列
        time_col = next((c for c in all_cols if 'Time' in c or '时间' in c), None)
        if not time_col: 
            return [] # 没时间无法分析
            
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        
        # === 构建矩阵 (Vectorized) ===
        N, M = len(df), len(heights)
        u_mat = np.full((N, M), np.nan, dtype=np.float32)
        std_mat = np.full((N, M), np.nan, dtype=np.float32)
        
        for i, h in enumerate(heights):
            # 模糊匹配列名
            col_u = next((c for c in all_cols if f'{h}m' in c and keyword in c and '方向' not in c), None)
            if not col_u: col_u = next((c for c in all_cols if f'{h}m' in c and ('风速' in c or 'Speed' in c) and '方向' not in c), None)
            
            col_std = next((c for c in all_cols if f'{h}m' in c and ('偏差' in c or 'std' in c.lower())), None)
            
            if col_u:
                u_mat[:, i] = pd.to_numeric(df[col_u], errors='coerce').values
            if col_std:
                std_mat[:, i] = pd.to_numeric(df[col_std], errors='coerce').values

        # === 矢量化筛选 ===
        u_mat[(u_mat <= 0) | (u_mat > 100)] = np.nan
        temp_u = np.nan_to_num(u_mat, nan=-1)
        max_idx = np.argmax(temp_u, axis=1)
        
        row_indices = np.arange(N)
        u_max = u_mat[row_indices, max_idx]
        z_max = np.array(heights)[max_idx]
        
        # TI
        std_val = std_mat[row_indices, max_idx]
        ti_val = np.divide(std_val, u_max, out=np.zeros_like(u_max), where=u_max!=0)
        
        # 降温
        temp_drop = np.zeros(N)
        temp_col = next((c for c in all_cols if '外温' in c or 'Temp' in c), None)
        if temp_col:
            t = pd.to_numeric(df[temp_col], errors='coerce').values
            # diff(1) = now - prev. Drop means negative diff.
            temp_drop = np.diff(t, prepend=t[0])
        
        # 判据
        cond_basic = (u_max > CONFIG['min_speed']) & (z_max <= CONFIG['max_nose_height'])
        cond_shape = (max_idx > 0) & (max_idx < M-1)
        cond_ti = ti_val > CONFIG['min_turbulence']
        cond_physics = (temp_drop < CONFIG['temp_drop_threshold']) | (u_max > 14.0)
        
        mask = cond_basic & cond_shape & cond_ti & cond_physics
        indices = np.where(mask)[0]
        
        # === Wood-Kwok 精修 ===
        final_events = []
        for idx in indices:
            row_u = u_mat[idx, :]
            passed, wk_params, r2 = verify_candidate_with_physics(heights, row_u)
            
            if passed:
                info = {
                    'station': station,
                    'timestamp': df.iloc[idx][time_col],
                    'u_max': float(u_max[idx]),
                    'z_max': int(z_max[idx]),
                    'ti': float(ti_val[idx]),
                    'temp_drop': float(temp_drop[idx]),
                    'wk_delta': float(wk_params['delta']),
                    'wk_r2': float(wk_params['r2'])
                }
                final_events.append(info)
                plot_verification(station, info['timestamp'], heights, row_u, wk_params, info)
        
        logger.info(f"  -> Found {len(final_events)} verified events.")
        return final_events

    except Exception as e:
        logger.error(f"Error in {station}: {e}")
        return []

# ================= 5. 主程序 =================
def main():
    logger.info("=== STARTING V4 ===")
    files = glob.glob(os.path.join(DATA_PATH, '*.txt'))
    
    all_results = []
    for f in files:
        all_results.extend(process_file(f))
        
    if all_results:
        res_df = pd.DataFrame(all_results)
        cols = ['station', 'timestamp', 'u_max', 'z_max', 'wk_delta', 'wk_r2', 'temp_drop', 'ti']
        res_df = res_df[cols].sort_values(['station', 'timestamp'])
        res_df.to_csv(CSV_OUTPUT, index=False, encoding='utf-8-sig')
        logger.info(f"Done. Saved to {CSV_OUTPUT}")
    else:
        logger.info("No events found.")

if __name__ == '__main__':
    main()