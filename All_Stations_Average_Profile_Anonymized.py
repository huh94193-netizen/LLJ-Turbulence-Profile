import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import os
import glob
import warnings

# ================= 配置区域 =================
# 数据路径
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
# 结果输出路径
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/all_stations_average_profiles_anonymized'

# 场站名称映射表 (关键字 -> 代号)
# 只要文件名包含左边的词，就会被重命名为右边的代号
STATION_MAPPING = {
    '大庆': '场站 1',
    '双鸭山': '场站 2',
    '昌图': '场站 3',   # 对应 铁岭昌图
    '台安': '场站 4',   # 对应 铁岭台安
    '衡水': '场站 5',
    '潍坊': '场站 6',
    '开封': '场站 7',
    '菏泽': '场站 8',
    '盐城': '场站 9',
    '孝感': '场站 10'
}

# LLJ 筛选标准
LLJ_THRESHOLD = 2.0       
MIN_JET_HEIGHT = 60       
MAX_JET_HEIGHT = 480      
MIN_WS_FOR_TI = 3.0       
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'details_per_station'), exist_ok=True)
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

# --- 1. 辅助工具 ---
def get_station_code(filename):
    """根据文件名关键字返回代号"""
    base = os.path.basename(filename)
    for key, code in STATION_MAPPING.items():
        if key in base:
            return code
    # 如果没匹配上，返回原名
    return base.split('-')[0]

def vector_mean_direction(deg_array):
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

# --- 2. 单站处理逻辑 ---
def process_station(file_path):
    # 获取代号
    station_code = get_station_code(file_path)
    print(f"处理文件: {os.path.basename(file_path)} -> 映射为: [{station_code}]")

    df = strict_tab_parse(file_path)
    if df is None: return None
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    # 识别高度
    clean_ws_cols = [c for c in df.columns if ('水平风速' in c or 'Speed' in c) and is_clean_col(c)]
    heights = sorted(list(set([int(re.search(r'(\d+)m', c).group(1)) for c in clean_ws_cols if re.search(r'(\d+)m', c)])))
    
    if len(heights) < 5: return None
    
    # 构建数据矩阵
    n_samples = len(df)
    n_heights = len(heights)
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

    # 筛选 LLJ
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
    if n_events < 10: return None

    # 计算统计量
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

    # 绘制单站详情图
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    axes[0].errorbar(mean_ws, heights, xerr=std_ws, fmt='-o', capsize=4, color='#1f77b4', label='Mean')
    axes[0].set_title(f'Mean Wind Speed (N={n_events})')
    axes[0].set_xlabel('m/s')
    axes[0].set_ylabel('Height [m]')
    
    axes[1].plot(mean_wd, heights, '-o', color='#2ca02c')
    axes[1].set_title('Mean Wind Direction')
    axes[1].set_xlabel('Degree')
    axes[1].set_xlim(0, 360)
    
    axes[2].errorbar(mean_ti, heights, xerr=std_ti, fmt='-o', capsize=4, color='#d62728')
    axes[2].set_title('Mean Turbulence Intensity')
    axes[2].set_xlabel('TI [-]')
    
    plt.suptitle(f"{station_code} - LLJ Profiles", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'details_per_station', f'{station_code}_Profile.png'))
    plt.close()

    return {
        'name': station_code,
        'heights': heights,
        'mean_ws': mean_ws,
        'mean_ti': mean_ti,
        'mean_wd': mean_wd,
        'n_samples': n_events
    }

# --- 3. 主程序 ---
def main():
    print("="*60)
    print(" LLJ 平均廓线生成 (已应用场站代号映射)")
    print("="*60)
    
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    all_data = []

    for f in files:
        res = process_station(f)
        if res: all_data.append(res)
        
    if not all_data:
        print("未生成有效数据。")
        return

    # --- 关键：按代号排序 (Station 1, Station 2, ..., Station 10) ---
    def sort_key(item):
        name = item['name']
        # 提取数字进行排序
        match = re.search(r'(\d+)', name)
        if match:
            return int(match.group(1))
        return 999 # 没数字的排最后

    all_data.sort(key=sort_key)

    print("\n>>> 正在生成总览对比图 (按 Station 1-10 排序) ...")
    
    # 使用 Jet 颜色映射，保证10条线颜色区分明显
    colors = plt.cm.jet(np.linspace(0, 1, len(all_data)))
    
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

    # 修饰图表
    axes[0].set_title('Mean Wind Speed', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Wind Speed [m/s]')
    axes[0].set_ylabel('Height [m]')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    # 图例
    axes[0].legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1))

    axes[1].set_title('Mean Wind Direction', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Direction [°]')
    axes[1].set_xlim(0, 360)
    axes[1].set_xticks(np.arange(0, 361, 90))
    axes[1].grid(True, linestyle='--', alpha=0.5)

    axes[2].set_title('Mean TI', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('TI [-]')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Low-Level Jet Average Profiles Comparison (All Stations Anonymized)', fontsize=20, y=0.98)
    plt.tight_layout()
    
    out_all = os.path.join(OUTPUT_DIR, 'All_Stations_Comparison_Anonymized.png')
    plt.savefig(out_all, dpi=300)
    print(f"[完成] 对比图已保存: {out_all}")

if __name__ == "__main__":
    main()