import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import os
import warnings

# ================= 用户配置区域 =================
# 1. 数据文件路径 (双鸭山集贤)
FILE_PATH = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
# 2. 输出目录
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/shuangyashan_analysis'

# 3. 急流判定标准
LLJ_THRESHOLD = 2.0       # 鼻尖风速需比上下边界大 2m/s
MIN_JET_HEIGHT = 60       # 急流鼻尖高度下限
MAX_JET_HEIGHT = 480      # 急流鼻尖高度上限

# 4. 质量控制
MIN_WS_FOR_TI = 3.0       # 计算 TI 时，风速小于 3m/s 的数据剔除 (防止 TI 虚高)
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 0. 字体配置 (防止乱码) ---
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
    """读取 Windographer 导出文件"""
    encodings = ['utf-8', 'gbk', 'gb18030', 'latin-1']
    for enc in encodings:
        try:
            # 尝试跳过前12行读取
            df = pd.read_csv(file_path, sep='\t', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(df.columns): return df
            # 尝试空格分隔
            df = pd.read_csv(file_path, sep='\s+', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(df.columns): return df
        except: continue
    return None

def is_clean_col(col_name):
    blacklist = ['最大', '最小', '偏差', '矢量', '标准差', 'Max', 'Min', 'Std', 'Gust']
    for bad in blacklist:
        if bad.lower() in col_name.lower(): return False
    return True

# --- 2. 主分析逻辑 ---
def analyze_profiles():
    print(f"正在读取文件: {os.path.basename(FILE_PATH)} ...")
    df = strict_tab_parse(FILE_PATH)
    if df is None:
        print("[Error] 读取失败，请检查文件路径或格式。")
        return

    # 清洗列名
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    # 1. 识别高度层
    clean_ws_cols = [c for c in df.columns if ('水平风速' in c or 'Speed' in c) and is_clean_col(c)]
    heights = sorted(list(set([int(re.search(r'(\d+)m', c).group(1)) for c in clean_ws_cols if re.search(r'(\d+)m', c)])))
    print(f"识别高度层: {heights}")
    
    n_samples = len(df)
    n_heights = len(heights)

    # 2. 构建数据矩阵
    mat_ws = np.full((n_samples, n_heights), np.nan) # 风速
    mat_wd = np.full((n_samples, n_heights), np.nan) # 风向
    mat_ti = np.full((n_samples, n_heights), np.nan) # 湍流

    for i, h in enumerate(heights):
        # 找列名
        col_ws = next((c for c in df.columns if f'{h}m' in c and '水平风速' in c and is_clean_col(c)), None)
        col_wd = next((c for c in df.columns if f'{h}m' in c and '风向' in c and is_clean_col(c)), None)
        col_std = next((c for c in df.columns if f'{h}m' in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)

        if col_ws: mat_ws[:, i] = pd.to_numeric(df[col_ws], errors='coerce').values
        if col_wd: mat_wd[:, i] = pd.to_numeric(df[col_wd], errors='coerce').values
        
        # 计算 TI = Std / Mean
        if col_std and col_ws:
            val_std = pd.to_numeric(df[col_std], errors='coerce').values
            val_ws = mat_ws[:, i]
            with np.errstate(divide='ignore', invalid='ignore'):
                val_ti = val_std / val_ws
                val_ti[val_ws < MIN_WS_FOR_TI] = np.nan # 剔除小风速
            mat_ti[:, i] = val_ti

    # 3. 筛选 LLJ 事件
    print("正在筛选 LLJ 事件...")
    llj_indices = []
    
    for i in range(n_samples):
        u = mat_ws[i, :]
        if np.isnan(u).any(): continue # 简单起见，只用完整数据
        
        idx_max = np.argmax(u)
        u_jet = u[idx_max]
        z_jet = heights[idx_max]
        
        # 判定高度范围
        if not (MIN_JET_HEIGHT <= z_jet <= MAX_JET_HEIGHT): continue
        # 排除边界极值
        if idx_max == 0 or idx_max == n_heights - 1: continue
        
        # 判定切变强度
        shear_bottom = u_jet - u[0]
        shear_top = u_jet - u[-1]
        
        if (shear_bottom >= LLJ_THRESHOLD) and (shear_top >= LLJ_THRESHOLD):
            llj_indices.append(i)

    n_events = len(llj_indices)
    print(f"共筛选出 {n_events} 个低空急流样本。")
    
    if n_events < 5:
        print("样本太少，无法统计。")
        return

    # 4. 计算统计廓线 (Mean & Std)
    # 提取急流子集
    llj_ws = mat_ws[llj_indices, :]
    llj_wd = mat_wd[llj_indices, :]
    llj_ti = mat_ti[llj_indices, :]
    
    # 计算统计量
    profile_ws_mean = np.nanmean(llj_ws, axis=0)
    profile_ws_std  = np.nanstd(llj_ws, axis=0)
    
    profile_ti_mean = np.nanmean(llj_ti, axis=0)
    profile_ti_std  = np.nanstd(llj_ti, axis=0)
    
    profile_wd_mean = []
    profile_wd_std = []
    for i in range(n_heights):
        wd_slice = llj_wd[:, i]
        profile_wd_mean.append(vector_mean_direction(wd_slice))
        # 风向标准差 (简单标量计算，仅作离散度参考)
        profile_wd_std.append(np.nanstd(wd_slice))

    # 5. 绘图 (带误差棒)
    print("正在绘图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    
    # 颜色设置
    color_ws = '#1f77b4' # 蓝
    color_wd = '#2ca02c' # 绿
    color_ti = '#d62728' # 红

    # --- Plot 1: 风速 ---
    ax = axes[0]
    ax.errorbar(profile_ws_mean, heights, xerr=profile_ws_std, 
                fmt='-o', capsize=4, color=color_ws, ecolor='gray', elinewidth=1, 
                label='Mean ± Std')
    ax.set_title('Wind Speed Profile', fontweight='bold')
    ax.set_xlabel('Wind Speed [m/s]')
    ax.set_ylabel('Height [m]')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    # --- Plot 2: 风向 ---
    ax = axes[1]
    ax.errorbar(profile_wd_mean, heights, xerr=profile_wd_std, 
                fmt='-o', capsize=4, color=color_wd, ecolor='gray', elinewidth=1,
                label='Vector Mean ± Std')
    ax.set_title('Wind Direction Profile', fontweight='bold')
    ax.set_xlabel('Direction [°]')
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    # --- Plot 3: 湍流强度 (TI) ---
    ax = axes[2]
    ax.errorbar(profile_ti_mean, heights, xerr=profile_ti_std, 
                fmt='-o', capsize=4, color=color_ti, ecolor='gray', elinewidth=1,
                label='Mean ± Std')
    ax.set_title('Turbulence Intensity (TI) Profile', fontweight='bold')
    ax.set_xlabel('TI [-]')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    # 全局标题
    plt.suptitle(f'双鸭山集贤 LLJ Average Profiles (Samples: {n_events})\nError Bars represent ±1 Standard Deviation', fontsize=16)
    
    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, 'Shuangyashan_LLJ_Profiles_Errorbar.png')
    plt.savefig(out_img, dpi=300)
    print(f"[完成] 图片已保存: {out_img}")
    
    # 保存 CSV 数据
    out_csv = os.path.join(OUTPUT_DIR, 'Shuangyashan_LLJ_Stats.csv')
    df_res = pd.DataFrame({
        'Height_m': heights,
        'WS_Mean': profile_ws_mean, 'WS_Std': profile_ws_std,
        'WD_Mean': profile_wd_mean, 'WD_Std': profile_wd_std,
        'TI_Mean': profile_ti_mean, 'TI_Std': profile_ti_std
    })
    df_res.to_csv(out_csv, index=False)
    print(f"[完成] 数据已保存: {out_csv}")

if __name__ == "__main__":
    analyze_profiles()