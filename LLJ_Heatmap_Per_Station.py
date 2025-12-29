import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import glob
import warnings

# ================= 配置区域 =================
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/lookup_table/stations'
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 1. 绘图函数 ---
def plot_heatmap(df_pivot, title, filename):
    plt.figure(figsize=(10, 6))
    # 颜色越深代表概率越高，使用 'RdYlBu_r' (红-黄-蓝反转，红色为高)
    sns.heatmap(df_pivot, annot=True, fmt=".0f", cmap='RdYlBu_r', 
                linewidths=.5, vmin=0, vmax=100)
    plt.title(title, fontsize=14)
    plt.xlabel('Hub Wind Speed Bin [m/s]')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# --- 2. 稳健的数据读取函数 (修复版) ---
def read_site_data(file_path):
    station_name = os.path.basename(file_path)
    
    # A. 尝试多种编码和分隔符读取
    df = None
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16', 'latin-1']
    
    for enc in encodings:
        try:
            # 尝试1: 跳过12行，Tab分隔
            temp = pd.read_csv(file_path, sep='\t', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp.columns):
                df = temp
                break
            # 尝试2: 跳过12行，空格分隔
            temp = pd.read_csv(file_path, sep='\s+', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp.columns):
                df = temp
                break
        except: continue
        
    if df is None:
        print(f"  [Error] 读取失败: {station_name}")
        return None # 返回单个 None

    # 清洗列名
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]

    # B. 提取时间
    try:
        # 优先找 Date/Time
        time_col = [c for c in df.columns if 'Date' in c or 'Time' in c][0]
        df['dt'] = pd.to_datetime(df[time_col], errors='coerce')
        df['Hour'] = df['dt'].dt.hour
    except:
        print(f"  [Error] 时间解析失败: {station_name}")
        return None

    # C. 提取 120m 轮毂风速 (或最近高度)
    cols_120 = [c for c in df.columns if '120m水平风速' in c and '最大' not in c]
    if cols_120:
        hub_col = cols_120[0]
    else:
        # 找最近高度
        speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c]
        if not speed_cols: return None
        heights = [int(re.search(r'(\d+)', c).group(1)) for c in speed_cols]
        closest_h = min(heights, key=lambda x: abs(x - 120))
        hub_col = [c for c in df.columns if f'{closest_h}m水平风速' in c and '最大' not in c][0]
        
    df['Hub_WS'] = pd.to_numeric(df[hub_col], errors='coerce')
    
    # D. 提取全高度层数据 (用于判定急流)
    speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c and '偏差' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    n_samples = len(df)
    n_heights = len(heights)
    ws_mat = np.full((n_samples, n_heights), np.nan)
    
    for i, h in enumerate(heights):
        col = [c for c in df.columns if f'{h}m水平风速' in c and '最大' not in c][0]
        ws_mat[:, i] = pd.to_numeric(df[col], errors='coerce').values
        
    # 返回: DataFrame, (风速矩阵, 高度列表)
    return df, (ws_mat, heights)

# --- 3. 主程序 ---
def main():
    print(f"开始生成分场站热力图...")
    print(f"输出目录: {OUTPUT_DIR}")
    
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    if not files:
        print("未找到数据文件！")
        return

    # 全局工况分箱设置
    WS_BINS = [3, 5, 7, 9, 11, 13, 15, 25]
    WS_LABELS = ['3-5', '5-7', '7-9', '9-11', '11-13', '13-15', '>15']

    success_count = 0

    for f in files:
        station_name = os.path.basename(f).split('-')[0]
        print(f"Processing: {station_name} ...", end='')
        
        # 1. 安全读取 (Safe Call)
        result = read_site_data(f)
        
        # 2. 空值检查 (防止 TypeError)
        if result is None:
            print(" [跳过] 读取返回空")
            continue
            
        df, (ws_mat, heights) = result
        print(" 成功")

        # 3. 判定 LLJ
        # 有效数据掩码
        valid_mask = ~np.isnan(ws_mat).any(axis=1) & df['Hub_WS'].notna() & df['Hour'].notna()
        
        ws_valid = ws_mat[valid_mask]
        if len(ws_valid) == 0: continue
        
        # 向量化判定
        max_idx = np.argmax(ws_valid, axis=1)
        u_jets = ws_valid[np.arange(len(ws_valid)), max_idx]
        z_jets = np.array(heights)[max_idx]
        
        # 判定条件: 高度 60-480m, 上下切变 > 2.0 m/s
        is_llj_vals = (z_jets >= 60) & (z_jets <= 480) & \
                      ((u_jets - ws_valid[:,0]) >= 2.0) & ((u_jets - ws_valid[:,-1]) >= 2.0)
        
        # 填回 DataFrame
        df['is_LLJ'] = False # 初始化全为 False
        # 仅在有效行位置填入结果 (注意对齐)
        # 这里的技巧是：df.loc[mask, col] = values
        df.loc[valid_mask, 'is_LLJ'] = is_llj_vals
        
        # 4. 分箱统计
        df['WS_Bin'] = pd.cut(df['Hub_WS'], bins=WS_BINS, labels=WS_LABELS)
        
        # 透视表: 计算概率 (mean * 100)
        pivot = df.pivot_table(index='Hour', columns='WS_Bin', values='is_LLJ', aggfunc='mean') * 100
        
        # 5. 绘图
        if not pivot.empty:
            out_name = os.path.join(OUTPUT_DIR, f'{station_name}_Prob_Map.png')
            plot_heatmap(pivot, f"{station_name} - LLJ Probability (%)", out_name)
            success_count += 1

    print("-" * 30)
    print(f"全部完成！共生成 {success_count} 张热力图。")

if __name__ == "__main__":
    main()