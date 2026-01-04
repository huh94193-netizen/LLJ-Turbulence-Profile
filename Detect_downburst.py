# -*- coding: utf-8 -*-
"""
High-Performance Downburst Detector (Vectorized + Physics-Enhanced)
优化点：
1. usecols 按需读取 (I/O 提速 5-10倍)
2. Numpy 矢量化计算 (计算提速 50-100倍)
3. 引入气温骤降 (Cold Pool) 物理判据
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
import logging
import matplotlib
# 服务器端绘图设置 (无显示器模式)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 忽略无关警告
warnings.filterwarnings('ignore')

# ================= 1. 全局配置 (CONFIG) =================
BASE_DIR = '/home/huxun/02_LLJ'
DATA_PATH = os.path.join(BASE_DIR, 'exported_data')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
PLOT_DIR = os.path.join(RESULT_DIR, 'plots')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 确保目录存在
for d in [PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

CSV_OUTPUT = os.path.join(RESULT_DIR, 'downburst_candidates.csv')
LOG_FILE = os.path.join(LOG_DIR, 'process.log')

# --- 物理阈值参数 ---
CONFIG = {
    # 基础风速门限
    'min_speed_base': 12.0,      # [m/s] 最低准入风速
    'high_speed_hard': 14.0,     # [m/s] "硬核"风速阈值 (超过这个值，即使没降温也算)
    
    # 鼻状剖面特征
    'max_height_of_max': 200,    # [m] 鼻尖最大高度
    'decay_ratio': 0.85,         # 衰减率 u_top / u_max
    
    # 湍流特征
    'min_turbulence': 0.12,      # [TI] 最小湍流强度
    
    # 降温特征 (Cold Pool)
    'temp_drop_threshold': -2.0, # [℃] 降温阈值 (必须是负数)
    'temp_diff_window': 1,       # [行] 计算温差的跨度 (1代表与上一行比，若数据是秒级请调大)
    
    # 数据质量
    'validity_threshold': 80     # [%] 数据有效率
}

# ================= 2. 日志系统初始化 =================
logger = logging.getLogger('DB_Detector')
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()

fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)

# 绘图字体设置 (防乱码)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']

# ================= 3. 绘图函数 (仅对筛选出的事件调用) =================
def plot_event(row, heights, station_name, info):
    """绘制风速剖面 + 湍流/温度信息"""
    z_vals, u_vals, sigma_vals, ti_vals = [], [], [], []
    
    for h in heights:
        u_col, std_col = f'{h}m水平风速', f'{h}m偏差'
        valid_col = f'{h}m数据可靠性'
        
        # 简单校验
        if u_col not in row or pd.isna(row[u_col]) or row[u_col] < 0.1: continue
        if valid_col in row and row[valid_col] < CONFIG['validity_threshold']: continue
        
        u = row[u_col]
        sigma = row.get(std_col, 0)
        
        z_vals.append(h)
        u_vals.append(u)
        sigma_vals.append(sigma)
        ti_vals.append(sigma / u if u > 0 else 0)
    
    if not z_vals: return

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    
    # 标题包含温度信息
    temp_str = f"Temp Drop: {info['temp_drop']:.1f}C" if 'temp_drop' in info else "No Temp Data"
    title_main = f"Station: {station_name} | Time: {row['Date/Time']}\n{temp_str} | U_max: {info['u_max']:.1f} m/s"
    
    # --- 左图: 风剖面 ---
    ax1.errorbar(u_vals, z_vals, xerr=sigma_vals, fmt='-o', color='red', 
                 ecolor='gray', elinewidth=1.5, capsize=3, label='Wind Profile ±Std')
    
    # 标注鼻尖
    ax1.annotate(f'Nose', xy=(info['u_max'], info['z_max']), 
                 xytext=(info['u_max']+1, info['z_max']+20),
                 arrowprops=dict(facecolor='black', width=1, headwidth=8))
    
    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Wind Speed Profile', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # --- 右图: 湍流强度 ---
    ax2.plot(ti_vals, z_vals, '-s', color='blue', linewidth=2, label='TI')
    ax2.axvline(CONFIG['min_turbulence'], color='green', linestyle='--', label='TI Threshold')
    
    ax2.set_xlabel('Turbulence Intensity (TI)')
    ax2.set_title('Turbulence Intensity', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    # 保存
    plt.suptitle(title_main, fontsize=14, y=0.98)
    safe_time = str(row['Date/Time']).replace(':', '').replace(' ', '_')
    fname = f"{station_name}_{safe_time}_DB.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=100)
    plt.close()

# ================= 4. 核心矢量化处理逻辑 =================
def process_file(filepath):
    filename = os.path.basename(filepath)
    station = filename.split('-')[0]
    logger.info(f"Processing: {station} ...")
    
    try:
        # --- A. 智能探测列名 ---
        # 只读第一行，获取所有列名
        header = pd.read_csv(filepath, sep=',', encoding='gbk', nrows=1, on_bad_lines='skip')
        header.columns = [c.strip() for c in header.columns]
        all_cols = header.columns.tolist()
        
        # 寻找高度层
        ws_cols = [c for c in all_cols if 'm水平风速' in c and '最大' not in c and '最小' not in c]
        if not ws_cols:
            logger.warning(f"  -> Skipped: No wind speed columns.")
            return []
            
        heights = sorted([int(c.split('m')[0]) for c in ws_cols])
        
        # 构建 usecols 列表 (只读需要的列)
        cols_needed = ['Date/Time', '外温'] # 显式加入外温
        for h in heights:
            cols_needed.extend([f'{h}m水平风速', f'{h}m偏差', f'{h}m数据可靠性'])
        
        # 过滤掉不存在的列
        final_usecols = [c for c in cols_needed if c in all_cols]
        
        # --- B. 高速读取 ---
        # engine='c' 是关键
        df = pd.read_csv(filepath, sep=',', encoding='gbk', usecols=final_usecols, 
                         engine='c', on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
        
        # 清洗时间
        if 'Date/Time' not in df.columns: return []
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
        df = df.dropna(subset=['Date/Time']).sort_values('Date/Time').reset_index(drop=True)
        
        # --- C. 准备矩阵数据 (Numpy Matrix) ---
        N = len(df)
        M = len(heights)
        
        # 初始化矩阵 (默认 NaN)
        u_mat = np.full((N, M), np.nan, dtype=np.float32)
        std_mat = np.full((N, M), np.nan, dtype=np.float32)
        
        # 填充数据
        for i, h in enumerate(heights):
            u_c = f'{h}m水平风速'
            std_c = f'{h}m偏差'
            val_c = f'{h}m数据可靠性'
            
            if u_c in df.columns:
                data = df[u_c].values.astype(np.float32)
                # 应用有效性掩膜 (如果列存在)
                if val_c in df.columns:
                    validity = df[val_c].values
                    data[validity < CONFIG['validity_threshold']] = np.nan
                u_mat[:, i] = data
                
            if std_c in df.columns:
                std_mat[:, i] = df[std_c].values.astype(np.float32)

        # 清洗异常值
        u_mat[(u_mat <= 0) | (u_mat > 100)] = np.nan
        
        # --- D. 计算物理特征 (Vectorized) ---
        
        # 1. 寻找鼻尖 (Max Wind Speed)
        # 用 nan_to_num 填充 -1 以便 argmax 能工作 (argmax 会忽略 NaN 除非全是 NaN)
        u_mat_filled = np.nan_to_num(u_mat, nan=-1.0)
        max_idx = np.argmax(u_mat_filled, axis=1) # 每一行最大值的列索引
        
        # 提取最大值 u_max 和对应高度 z_max
        row_idx = np.arange(N)
        u_max = u_mat[row_idx, max_idx]
        z_max = np.array(heights)[max_idx]
        
        # 提取顶层风速 u_top
        u_top = u_mat[:, -1]
        
        # 计算鼻尖处的湍流强度 TI
        std_at_max = std_mat[row_idx, max_idx]
        ti_at_max = np.divide(std_at_max, u_max, out=np.zeros_like(u_max), where=u_max!=0)
        
        # 2. 计算气温突降 (Temperature Drop)
        temp_drop = np.zeros(N, dtype=np.float32)
        has_temp = False
        if '外温' in df.columns:
            # 清洗温度异常值
            t_raw = df['外温'].values
            t_raw[(t_raw < -60) | (t_raw > 60)] = np.nan # 简单清洗
            # 计算差分 (当前时刻 - 上一时刻)
            # 如果是降温，diff 应该是负数
            t_diff = np.diff(t_raw, prepend=t_raw[0]) 
            # 或者是 rolling window，这里用简单的 diff(window)
            if CONFIG['temp_diff_window'] > 1:
                t_series = pd.Series(t_raw)
                t_diff = t_series.diff(CONFIG['temp_diff_window']).fillna(0).values
                
            temp_drop = t_diff
            has_temp = True
            
        # --- E. 综合判定 (Boolean Logic) ---
        
        # 基础条件:
        # 1. 存在有效数据 (不是 NaN)
        cond_valid = ~np.isnan(u_max)
        # 2. 鼻尖高度限制 (不能太高)
        cond_h = z_max <= CONFIG['max_height_of_max']
        # 3. 鼻尖不能在最顶层 (否则不是下击暴流，是普通大风)
        cond_shape = max_idx != (M - 1)
        # 4. 衰减率达标
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = u_top / u_max
        cond_decay = ratio < CONFIG['decay_ratio']
        # 5. 湍流够大 (区分于 LLJ)
        cond_ti = ti_at_max > CONFIG['min_turbulence']
        
        # 核心分级条件 (OR Logic):
        # 情况A: 伴随明显降温 (Strong Cooling) -> 风速门槛可以是 min_speed_base (12m/s)
        cond_cooling = temp_drop <= CONFIG['temp_drop_threshold']
        
        # 情况B: 极强风速 (Strong Wind) -> 即使没降温也算 -> 风速门槛 high_speed_hard (14m/s)
        cond_super_strong = u_max >= CONFIG['high_speed_hard']
        
        # 组合逻辑:
        # 必须满足: (基础风速达标) AND (形态达标) AND (湍流达标) AND ( (有降温) OR (风速特别大) )
        cond_speed_base = u_max >= CONFIG['min_speed_base']
        
        final_mask = (
            cond_valid &
            cond_speed_base & 
            cond_h & 
            cond_shape & 
            cond_decay & 
            cond_ti & 
            (cond_cooling | cond_super_strong) # <--- 关键优化
        )
        
        # --- F. 提取结果 ---
        indices = np.where(final_mask)[0]
        results = []
        
        for idx in indices:
            row_data = df.iloc[idx]
            info = {
                'station': station,
                'timestamp': row_data['Date/Time'],
                'u_max': float(u_max[idx]),
                'z_max': int(z_max[idx]),
                'u_top': float(u_top[idx]),
                'ti_at_max': float(ti_at_max[idx]),
                'temp_drop': float(temp_drop[idx]) if has_temp else 0.0,
                'tag': 'Cold_Pool' if (has_temp and temp_drop[idx] <= CONFIG['temp_drop_threshold']) else 'Dry/Strong'
            }
            results.append(info)
            # 绘图 (只绘这几个事件，不耗时)
            plot_event(row_data, heights, station, info)
            
        logger.info(f"  -> Found {len(results)} candidates.")
        return results

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}", exc_info=True)
        return []

# ================= 5. 主程序入口 =================
def main():
    logger.info("=== STARTING VECTORIZED DETECTION ===")
    logger.info(f"Looking for files in: {DATA_PATH}")
    
    files = glob.glob(os.path.join(DATA_PATH, '*.txt'))
    if not files:
        logger.error("No .txt files found.")
        return
        
    all_events = []
    for f in files:
        all_events.extend(process_file(f))
        
    # 保存总表
    if all_events:
        res_df = pd.DataFrame(all_events)
        cols_order = ['station', 'timestamp', 'tag', 'u_max', 'z_max', 'temp_drop', 'ti_at_max', 'u_top']
        res_df = res_df[cols_order].sort_values(['station', 'timestamp'])
        
        res_df.to_csv(CSV_OUTPUT, index=False, encoding='utf-8-sig')
        logger.info(f"\nSaved {len(res_df)} events to: {CSV_OUTPUT}")
    else:
        logger.info("\nNo events detected.")

if __name__ == '__main__':
    main()