import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ================= 配置区域 =================
file_path = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'

# 1. 质量控制
MIN_AVAILABILITY = 80.0 

# 2. 急流判定
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

def strict_tab_parse_v3(file_path):
    print(f" -> 启动严格 Tab 解析模式 (v5.0 排除注释干扰)...")
    
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'latin-1']
    raw_lines = []
    
    # 1. 读取所有行
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                raw_lines = f.readlines()
            print(f" -> 使用 [{enc}] 编码读取成功，共 {len(raw_lines)} 行。")
            break
        except:
            continue
            
    if not raw_lines:
        print("[错误] 无法打开文件。")
        return None

    # 2. 寻找表头行 (核心修改)
    header_idx = -1
    header_parts = []
    
    for i, line in enumerate(raw_lines[:100]):
        # 【修改点】: 必须包含 "Date/Time" 这个特定组合，防止被 "Time stamps..." 注释误导
        # 或者包含 "m水平风速" (防止表头被改过)
        if "Date/Time" in line or "m水平风速" in line:
            header_idx = i
            header_parts = line.strip().split('\t')
            break
            
    if header_idx == -1:
        print("[错误] 找不到表头 (未发现 'Date/Time' 关键词)。")
        print(" -> 请检查文件前几行是否包含 'Date/Time'。")
        return None

    # 清洗表头
    header_parts = [h.strip().replace('"', '') for h in header_parts]
    print(f" -> 表头定位在第 {header_idx + 1} 行。")
    # 打印前几个列名确认一下
    print(f" -> 表头预览 (前5列): {header_parts[:5]}")
    
    if "Time stamps" in str(header_parts):
        print("[警告] 似乎还是读到了注释行，请检查文件格式！")

    # 3. 解析数据行
    data_list = []
    
    for i in range(header_idx + 1, len(raw_lines)):
        line = raw_lines[i].strip()
        if not line: continue
        
        parts = line.split('\t')
        parts = [p.strip().replace('"', '') for p in parts]
        
        # 对齐逻辑
        if len(parts) > len(header_parts):
            parts = parts[:len(header_parts)]
        elif len(parts) < len(header_parts):
            parts += [''] * (len(header_parts) - len(parts))
            
        if not parts[0]: 
            continue
            
        data_list.append(parts)

    if not data_list:
        print("[错误] 未提取到任何数据。")
        return None

    # 4. 生成 DataFrame
    df = pd.DataFrame(data_list, columns=header_parts)
    print(f" -> 解析成功！有效数据行数: {len(df)}")
    return df

def process_lidar_data(file_path):
    print("="*60)
    print(f"正在处理文件: {os.path.basename(file_path)}")
    
    df = strict_tab_parse_v3(file_path)
    
    if df is None:
        return

    # -------------------------------------------------------
    # 数据清洗
    # -------------------------------------------------------
    print(" -> 正在清洗列名...")
    df.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df.columns]
    
    speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c and '最小' not in c]
    
    if not speed_cols:
        print("[错误] 未找到 'm水平风速' 列。")
        print(f" -> 当前列名(前10个): {list(df.columns)[:10]}")
        return

    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    print(f" -> 识别到的高度层: {heights}")

    # -------------------------------------------------------
    # 计算 TI
    # -------------------------------------------------------
    print(" -> 正在计算每层湍流强度 (TI)...")
    
    avail_col_500 = '500m数据可靠性'
    if avail_col_500 in df.columns:
        df[avail_col_500] = pd.to_numeric(df[avail_col_500], errors='coerce')
        df = df[df[avail_col_500] >= MIN_AVAILABILITY].copy()
    
    ti_cols = []
    for h in heights:
        ws_col = f'{h}m水平风速'
        std_col = f'{h}m偏差' 
        ti_col_name = f'{h}m_TI'
        
        if ws_col in df.columns and std_col in df.columns:
            df[ws_col] = pd.to_numeric(df[ws_col], errors='coerce')
            df[std_col] = pd.to_numeric(df[std_col], errors='coerce')
            
            df[ti_col_name] = df[std_col] / df[ws_col]
            df.loc[df[ws_col] < 3.0, ti_col_name] = np.nan
            ti_cols.append(ti_col_name)

    # -------------------------------------------------------
    # 识别急流
    # -------------------------------------------------------
    print(" -> 正在筛选急流样本...")
    
    sorted_speed_cols = [f'{h}m水平风速' for h in heights]
    
    for col in sorted_speed_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df_clean = df.dropna(subset=sorted_speed_cols)
    speed_matrix = df_clean[sorted_speed_cols].values
    
    llj_indices = []
    
    if len(speed_matrix) == 0:
        print("[错误] 有效数据为空。")
        return

    for i in range(len(speed_matrix)):
        speeds = speed_matrix[i]
        
        try:
            max_idx = np.nanargmax(speeds)
        except:
            continue
            
        max_v = speeds[max_idx]
        max_h = heights[max_idx]
        
        if max_h <= MIN_JET_HEIGHT or max_h >= MAX_JET_HEIGHT:
            continue
            
        v_bottom = speeds[0]
        v_top = speeds[-1]
        
        if (max_v - v_bottom >= LLJ_THRESHOLD) and (max_v - v_top >= LLJ_THRESHOLD):
            llj_indices.append(df_clean.index[i])
            
    llj_df = df.loc[llj_indices]
    print(f" -> 筛选完成! 捕获急流样本数: {len(llj_df)} (占比 {len(llj_df)/len(df)*100:.2f}%)")

    if len(llj_df) < 5:
        print("[提示] 样本过少，不绘图。")
        return

    # -------------------------------------------------------
    # 绘图
    # -------------------------------------------------------
    mean_speed = llj_df[sorted_speed_cols].mean()
    mean_ti = llj_df[ti_cols].mean()
    
    fig, ax1 = plt.subplots(figsize=(9, 12))
    
    color_ws = '#1f77b4'
    ax1.set_xlabel('Wind Speed [m/s]', color=color_ws, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Height [m]', fontsize=14, fontweight='bold')
    line1 = ax1.plot(mean_speed.values, heights, color=color_ws, marker='o', 
                     linewidth=3, label='Wind Speed')
    ax1.tick_params(axis='x', labelcolor=color_ws)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twiny()
    color_ti = '#d62728'
    ax2.set_xlabel('TI [-]', color=color_ti, fontsize=14, fontweight='bold')
    line2 = ax2.plot(mean_ti.values, heights, color=color_ti, marker='s', 
                     linestyle='--', linewidth=3, label='TI')
    ax2.tick_params(axis='x', labelcolor=color_ti)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title(f'LLJ Profile (V5.0)\nEvents: {len(llj_df)}', fontsize=16)
    
    out_png = 'LLJ_Result_V5.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    
    out_csv = 'LLJ_Result_Data.csv'
    res = pd.DataFrame({'Height': heights, 'Speed': mean_speed.values, 'TI': mean_ti.values})
    res.to_csv(out_csv, index=False)
    
    print("="*60)
    print(f"[成功] 图片已保存: {os.path.abspath(out_png)}")
    print(f"[成功] 数据已保存: {os.path.abspath(out_csv)}")

if __name__ == "__main__":
    if os.path.exists(file_path):
        process_lidar_data(file_path)
    else:
        print(f"文件不存在: {file_path}")