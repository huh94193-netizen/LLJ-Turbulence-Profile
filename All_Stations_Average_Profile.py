import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import os
import glob
import warnings

# ================= 配置区域 =================
# 数据存放路径
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
# 结果输出路径
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/all_stations_average_profiles'

# 急流判定标准
LLJ_THRESHOLD = 2.0       # 鼻尖风速需比上下边界大 2m/s
MIN_JET_HEIGHT = 60       # 急流鼻尖高度下限
MAX_JET_HEIGHT = 480      # 急流鼻尖高度上限

# 质量控制
MIN_WS_FOR_TI = 3.0       # 计算 TI 时剔除小风速
# ===========================================

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'details_per_station'), exist_ok=True)
warnings.filterwarnings('ignore')

# --- 0. 字体智能配置 ---
def get_safe_font_prop():
    candidates = ['WenQuanYi Micro Hei', 'Zen Hei', 'Droid Sans Fallback', 'SimHei', 'Microsoft YaHei', 'SimSun']
    for font in candidates:
        try:
            if fm.findfont(font) != fm.findfont('DejaVu Sans'): return font
        except: continue
    return None

SYSTEM_ZH_FONT = get_safe_font_prop()
if SYSTEM_ZH_FONT:
    plt.rcParams['font.sans-serif'] = [SYSTEM_ZH_FONT]
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("[提示] 未检测到中文字体，将使用英文 ID 显示场站名。")

# --- 1. 辅助函数 ---
def vector_mean_direction(deg_array):
    """计算风向的矢量平均"""
    rads = np.radians(deg_array)
    sin_mean = np.nanmean(np.sin(rads))
    cos_mean = np.nanmean(np.cos(rads))
    mean_deg = np.degrees(np.arctan2(sin_mean, cos_mean))
    if mean_deg < 0: mean_deg += 360
    return mean_deg

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

def is_clean_col(col_name):
    blacklist = ['最大', '最小', '偏差', '矢量', '标准差', 'Max', 'Min', 'Std', 'Gust']
    for bad in blacklist:
        if bad.lower() in col_name.lower(): return False
    return True

# --- 2. 单场站处理逻辑 ---
def process_station(file_path):
    filename = os.path.basename(file_path)
    # 提取显示名称
    try:
        if SYSTEM_ZH_FONT:
            station_name = filename.split('-')[0] # 中文名
        else:
            station_name = filename.split('-')[1] # ID
    except:
        station_name = filename[:10]

    print(f"[{station_name}] 正在处理...", end=' ')

    df = strict_tab_parse(file_path)
    if df is None: 
        print("读取失败")
        return None
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    # 1. 识别高度层
    clean_ws_cols = [c for c in df.columns if ('水平风速' in c or 'Speed' in c) and is_clean_col(c)]
    heights = sorted(list(set([int(re.search(r'(\d+)m', c).group(1)) for c in clean_ws_cols if re.search(r'(\d+)m', c)])))
    
    if len(heights) < 5:
        print("高度层不足")
        return None

    n_samples = len(df)
    n_heights = len(heights)

    # 2. 构建矩阵
    mat_ws = np.full((n_samples, n_heights), np.nan)
    mat_wd = np.full((n_samples, n_heights), np.nan)
    mat_ti = np.full((n_samples, n_heights), np.nan)

    for i, h in enumerate(heights):
        col_ws = next((c for c in df.columns if f'{h}m' in c and '水平风速' in c and is_clean_col(c)), None)
        col_wd = next((c for c in df.columns if f'{h}m' in c and '风向' in c and is_clean_col(c)), None)
        col_std = next((c for c in df.columns if f'{h}m' in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)

        if col_ws: mat_ws[:, i] = pd.to_numeric(df[col_ws], errors='coerce').values
        if col_wd: mat_wd[:, i] = pd.to_numeric(df[col_wd], errors='coerce').values
        if col_std and col_ws:
            val_std = pd.to_numeric(df[col_std], errors='coerce').values
            val_ws = mat_ws[:, i]
            with np.errstate(divide='ignore', invalid='ignore'):
                val_ti = val_std / val_ws
                val_ti[val_ws < MIN_WS_FOR_TI] = np.nan
            mat_ti[:, i] = val_ti

    # 3. 筛选 LLJ
    llj_indices = []
    for i in range(n_samples):
        u = mat_ws[i, :]
        if np.isnan(u).any(): continue
        
        idx_max = np.argmax(u)
        u_jet = u[idx_max]
        z_jet = heights[idx_max]
        
        if not (MIN_JET_HEIGHT <= z_jet <= MAX_JET_HEIGHT): continue
        if idx_max == 0 or idx_max == n_heights - 1: continue
        
        if (u_jet - u[0] >= LLJ_THRESHOLD) and (u_jet - u[-1] >= LLJ_THRESHOLD):
            llj_indices.append(i)

    n_events = len(llj_indices)
    if n_events < 10:
        print(f"样本过少 ({n_events})")
        return None

    # 4. 计算统计量
    llj_ws = mat_ws[llj_indices, :]
    llj_wd = mat_wd[llj_indices, :]
    llj_ti = mat_ti[llj_indices, :]
    
    mean_ws = np.nanmean(llj_ws, axis=0)
    std_ws = np.nanstd(llj_ws, axis=0)
    
    mean_ti = np.nanmean(llj_ti, axis=0)
    std_ti = np.nanstd(llj_ti, axis=0)
    
    mean_wd = []
    for i in range(n_heights):
        mean_wd.append(vector_mean_direction(llj_wd[:, i]))
    mean_wd = np.array(mean_wd)

    print(f"完成 (样本数: {n_events})")

    # 5. 生成单站详情图 (带误差棒)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    
    # WS
    axes[0].errorbar(mean_ws, heights, xerr=std_ws, fmt='-o', capsize=4, color='#1f77b4', ecolor='gray', alpha=0.9)
    axes[0].set_title(f'Mean Wind Speed\n(Samples: {n_events})')
    axes[0].set_xlabel('m/s')
    axes[0].set_ylabel('Height [m]')
    axes[0].grid(True, ls=':')
    
    # WD
    axes[1].plot(mean_wd, heights, '-o', color='#2ca02c')
    axes[1].set_title('Mean Wind Direction')
    axes[1].set_xlabel('Degree')
    axes[1].set_xlim(0, 360)
    axes[1].set_xticks(np.arange(0, 361, 90))
    axes[1].grid(True, ls=':')
    
    # TI
    axes[2].errorbar(mean_ti, heights, xerr=std_ti, fmt='-o', capsize=4, color='#d62728', ecolor='gray', alpha=0.9)
    axes[2].set_title('Mean Turbulence Intensity')
    axes[2].set_xlabel('TI [-]')
    axes[2].grid(True, ls=':')
    
    plt.suptitle(f"{station_name} - LLJ Average Profiles (with Std Dev)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'details_per_station', f'{station_name}_Profile.png'), dpi=150)
    plt.close()

    # 返回数据供汇总
    return {
        'name': station_name,
        'heights': heights,
        'mean_ws': mean_ws,
        'mean_ti': mean_ti,
        'mean_wd': mean_wd,
        'n_samples': n_events
    }

# --- 3. 主程序：遍历与汇总绘图 ---
def main():
    print("="*60)
    print(" 全场站平均廓线生成程序 (All Stations Average Profile)")
    print("="*60)
    
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    all_data = []

    # 遍历处理
    for f in files:
        res = process_station(f)
        if res: all_data.append(res)
        
    if not all_data:
        print("未生成有效数据。")
        return

    print("\n>>> 正在生成总览对比图 (Comparison Maps) ...")
    
    # 颜色库
    colors = plt.cm.jet(np.linspace(0, 1, len(all_data)))
    
    # 绘图：三张大图 (WS对比, TI对比, WD对比)
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=True)
    
    for i, d in enumerate(all_data):
        h = d['heights']
        label_str = f"{d['name']} (N={d['n_samples']})"
        
        # WS
        axes[0].plot(d['mean_ws'], h, '-o', markersize=4, color=colors[i], label=label_str, linewidth=2, alpha=0.8)
        
        # WD
        axes[1].plot(d['mean_wd'], h, '-o', markersize=4, color=colors[i], linewidth=2, alpha=0.8)
        
        # TI
        axes[2].plot(d['mean_ti'], h, '-o', markersize=4, color=colors[i], linewidth=2, alpha=0.8)

    # 修饰 WS 图
    axes[0].set_title('All Stations: Mean Wind Speed', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Wind Speed [m/s]', fontsize=12)
    axes[0].set_ylabel('Height [m]', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=9, loc='upper left') # 图例只放第一张

    # 修饰 WD 图
    axes[1].set_title('All Stations: Mean Wind Direction', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Direction [°]', fontsize=12)
    axes[1].set_xlim(0, 360)
    axes[1].set_xticks(np.arange(0, 361, 90))
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 修饰 TI 图
    axes[2].set_title('All Stations: Mean TI', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('TI [-]', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Low-Level Jet Average Profiles Comparison (All Stations)', fontsize=20, y=0.98)
    plt.tight_layout()
    
    out_all = os.path.join(OUTPUT_DIR, 'All_Stations_Comparison.png')
    plt.savefig(out_all, dpi=300)
    print(f"[完成] 总览对比图已保存: {out_all}")
    print(f"[完成] 单站详情图已保存至: {os.path.join(OUTPUT_DIR, 'details_per_station')}")

if __name__ == "__main__":
    main()