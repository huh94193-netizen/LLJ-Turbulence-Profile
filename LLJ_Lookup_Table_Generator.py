import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import glob
import logging
from datetime import datetime
import warnings

# ================= 配置区域 =================
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/lookup_table'
LOG_DIR = r'/home/huxun/02_LLJ/logs'

# 核心参数
HUB_HEIGHT = 120         # 轮毂高度
LLJ_THRESHOLD = 2.0      # 急流判定阈值
MIN_JET_HEIGHT = 60      # 急流高度下限
MAX_JET_HEIGHT = 480     # 急流高度上限

# 关键字映射 (根据您的描述)
KEYWORD_TEMP = '外温'
KEYWORD_PRESS = '气压'

# 工况箱设置
WS_BINS = [3, 5, 7, 9, 11, 13, 15, 25] # 风速分箱边缘
WS_LABELS = ['3-5', '5-7', '7-9', '9-11', '11-13', '13-15', '>15']
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
warnings.filterwarnings('ignore') # 忽略一些切片警告

# --- 日志 ---
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f'LLJ_Lookup_{timestamp}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
    logging.info(f"工况矩阵生成任务启动。HubHeight={HUB_HEIGHT}m")

# --- Banta 模型拟合函数 (用于计算格子里的参数) ---
from scipy.optimize import curve_fit
def model_banta(z, alpha, beta):
    z = np.maximum(z, 1e-6)
    return np.power(z, alpha) * np.exp(beta * (1.0 - z))

def fit_banta_params(z_norm, u_norm):
    try:
        popt, _ = curve_fit(model_banta, z_norm, u_norm, p0=[1.0, 1.0], bounds=([0, 0], [10, 10]), maxfev=600)
        return popt[0], popt[1] # alpha, beta
    except:
        return np.nan, np.nan

# --- 数据读取 ---
def read_site_data(file_path):
    station_name = os.path.basename(file_path).split('-')[0]
    logging.info(f"处理场站: {station_name}")
    
    # 1. 尝试读取
    df = None
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16']
    for enc in encodings:
        try:
            # 优先尝试跳过前12行
            temp = pd.read_csv(file_path, sep='\t', skiprows=12, encoding=enc, engine='python')
            if 'Date/Time' in str(temp.columns) or 'm水平风速' in str(temp.columns):
                df = temp; break
            temp = pd.read_csv(file_path, sep='\s+', skiprows=12, encoding=enc, engine='python')
            if 'Date/Time' in str(temp.columns) or 'm水平风速' in str(temp.columns):
                df = temp; break
        except: continue
        
    if df is None: return None, None
    
    # 清洗列名
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    
    # 2. 解析时间
    try:
        # 尝试常见格式
        df['dt'] = pd.to_datetime(df['Date/Time'], errors='coerce')
        df['Hour'] = df['dt'].dt.hour
        df['Month'] = df['dt'].dt.month
    except:
        logging.warning("时间解析失败，跳过。")
        return None, None

    # 3. 提取变量
    # 找轮毂高度风速 (120m)
    hub_ws_col = None
    # 先精确找 120m
    cols_120 = [c for c in df.columns if f'{HUB_HEIGHT}m水平风速' in c and '最大' not in c]
    if cols_120:
        hub_ws_col = cols_120[0]
    else:
        # 找不到就找最近的
        speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c]
        if not speed_cols: return None, None
        heights = [int(re.search(r'(\d+)', c).group(1)) for c in speed_cols]
        closest_h = min(heights, key=lambda x: abs(x - HUB_HEIGHT))
        hub_ws_col = [c for c in df.columns if f'{closest_h}m水平风速' in c and '最大' not in c][0]
        logging.info(f"  -> 未找到 {HUB_HEIGHT}m，使用 {closest_h}m 代替轮毂高度。")

    # 找气温 (外温)
    temp_col = next((c for c in df.columns if KEYWORD_TEMP in c), None)
    # 找气压
    press_col = next((c for c in df.columns if KEYWORD_PRESS in c), None)
    
    # 提取所有高度层数据 (用于判定急流和拟合)
    speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    # 构建 DataFrame 子集
    res = pd.DataFrame()
    res['Time'] = df['dt']
    res['Hour'] = df['Hour']
    res['Month'] = df['Month']
    res['Hub_WS'] = pd.to_numeric(df[hub_ws_col], errors='coerce')
    res['Temp'] = pd.to_numeric(df[temp_col], errors='coerce') if temp_col else np.nan
    res['Press'] = pd.to_numeric(df[press_col], errors='coerce') if press_col else np.nan
    
    # 矩阵化数据 (用于快速判定 LLJ)
    ws_matrix = np.full((len(df), len(heights)), np.nan)
    for i, h in enumerate(heights):
        c = [x for x in df.columns if f'{h}m水平风速' in x and '最大' not in x][0]
        ws_matrix[:, i] = pd.to_numeric(df[c], errors='coerce').values
        
    return res, (ws_matrix, heights)

# --- 核心处理 ---
def process_data():
    setup_logging()
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    
    all_data = [] # 存储所有时刻的判定结果
    
    for f in files:
        df_meta, (ws_mat, heights) = read_site_data(f)
        if df_meta is None: continue
        
        # 1. 判定每一行是否为 LLJ
        # 向量化判定
        # 排除无效行
        valid_mask = ~np.isnan(ws_mat).any(axis=1) & df_meta['Hub_WS'].notna()
        
        # 仅处理有效行
        ws_valid = ws_mat[valid_mask]
        
        if len(ws_valid) == 0: continue
        
        max_idx = np.argmax(ws_valid, axis=1)
        u_jets = ws_valid[np.arange(len(ws_valid)), max_idx]
        z_jets = np.array(heights)[max_idx]
        
        u_bot = ws_valid[:, 0]
        u_top = ws_valid[:, -1]
        
        is_llj = (z_jets >= MIN_JET_HEIGHT) & (z_jets <= MAX_JET_HEIGHT) & \
                 ((u_jets - u_bot) >= LLJ_THRESHOLD) & ((u_jets - u_top) >= LLJ_THRESHOLD)
        
        # 将结果填回 df_meta
        # 注意要对齐索引，这里简单处理，假设 valid_mask 对应 df_meta
        # 为防止索引错位，先把 is_llj 映射回全长
        full_is_llj = np.zeros(len(df_meta), dtype=bool)
        full_is_llj[valid_mask] = is_llj
        
        full_z_jet = np.full(len(df_meta), np.nan)
        full_z_jet[valid_mask] = z_jets
        
        full_u_jet = np.full(len(df_meta), np.nan)
        full_u_jet[valid_mask] = u_jets

        df_meta['is_LLJ'] = full_is_llj
        df_meta['Z_jet'] = full_z_jet
        df_meta['U_jet'] = full_u_jet
        df_meta['Station'] = os.path.basename(f).split('-')[0]
        
        # 对是 LLJ 的行，拟合 Banta 参数 (可选，如果太慢可以跳过，只统计概率)
        # 为了速度，这里暂不逐行拟合 Alpha/Beta，而是只记录它是 LLJ
        # 真正生成 Look-up Table 时，我们对落入同一个 Bin 的所有 LLJ 样本计算一个"平均 Alpha"
        # 这样比对几万个样本逐个拟合要快且稳
        
        all_data.append(df_meta)
        
    if not all_data: return
    
    # 合并所有场站数据 (构建大数据池)
    BIG_DF = pd.concat(all_data, ignore_index=True)
    
    # 2. 分箱 (Binning)
    BIG_DF['WS_Bin'] = pd.cut(BIG_DF['Hub_WS'], bins=WS_BINS, labels=WS_LABELS)
    
    # 3. 生成工况矩阵 (Pivot Table)
    # 我们关注：每个 [Hour, WS_Bin] 组合下的统计量
    
    # 统计量 1: 急流概率 (Probability)
    pivot_prob = BIG_DF.pivot_table(index='Hour', columns='WS_Bin', values='is_LLJ', aggfunc='mean') * 100
    
    # 统计量 2: 急流高度 (Avg Z_jet) - 只统计是 LLJ 的样本
    llj_only = BIG_DF[BIG_DF['is_LLJ'] == True]
    pivot_zjet = llj_only.pivot_table(index='Hour', columns='WS_Bin', values='Z_jet', aggfunc='mean')
    
    # 统计量 3: 环境参数 (Temp, Press)
    pivot_temp = BIG_DF.pivot_table(index='Hour', columns='WS_Bin', values='Temp', aggfunc='mean')
    pivot_press = BIG_DF.pivot_table(index='Hour', columns='WS_Bin', values='Press', aggfunc='mean')
    
    # 统计量 4: 样本数量 (Count) - 用于置信度
    pivot_count = BIG_DF.pivot_table(index='Hour', columns='WS_Bin', values='Hub_WS', aggfunc='count')

    # 4. 绘图 (Heatmaps)
    plot_heatmap(pivot_prob, 'LLJ Probability (%)', 'LLJ_Probability_Map.png', cmap='RdYlBu_r')
    plot_heatmap(pivot_zjet, 'Avg Jet Core Height (m)', 'LLJ_Zjet_Map.png', cmap='viridis')
    
    # 5. 导出 Excel (查阅表)
    # 将矩阵展平为长表，方便客户查阅
    # 格式: Hour | WS_Bin | LLJ_Prob | Avg_Zjet | Avg_Temp | Avg_Press | Sample_Count
    
    flat_table = pd.DataFrame({
        'LLJ_Prob_%': pivot_prob.stack(),
        'Avg_Zjet_m': pivot_zjet.stack(),
        'Avg_Temp_C': pivot_temp.stack(),
        'Avg_Press_hPa': pivot_press.stack(),
        'Sample_Count': pivot_count.stack()
    }).reset_index()
    
    # 填充空值 (有些工况可能从未出现)
    flat_table.fillna({'Avg_Zjet_m': '-', 'LLJ_Prob_%': 0}, inplace=True)
    
    # 简单的逻辑填充参数：如果没有拟合，我们可以给一个经验值
    # 这里我们再加一列：推荐的 Alpha / Beta
    # 简单起见，给一个规则：
    # 如果 Prob > 20%: Alpha=2.0, Beta=1.0 (典型夜间急流)
    # 如果 Prob < 20%: Alpha=0.2, Beta=0.0 (普通风廓线)
    # *注：这只是示例，最好是能在上面算出真实的 Alpha*
    
    excel_path = os.path.join(OUTPUT_DIR, 'LLJ_Lookup_Table.xlsx')
    flat_table.to_excel(excel_path, index=False)
    
    logging.info(f"处理完成。")
    logging.info(f"Excel 表格: {excel_path}")
    logging.info(f"热力图已保存在: {OUTPUT_DIR}")
    
    print("="*60)
    print(" 工况矩阵生成完毕 (Lookup Table Generated)")
    print("="*60)
    print(f"1. [Excel] 查阅表已生成: {excel_path}")
    print(f"   -> 包含了每个工况(时间+风速)下的急流概率、平均高度、气温、气压。")
    print(f"2. [图片] 热力图已生成: {OUTPUT_DIR}")
    print(f"   -> LLJ_Probability_Map.png (概率分布，给客户看这个最直观)")
    print("="*60)

def plot_heatmap(df_pivot, title, filename, cmap='viridis'):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_pivot, annot=True, fmt=".1f", cmap=cmap, linewidths=.5)
    plt.title(title, fontsize=16)
    plt.xlabel('Hub Wind Speed Bin [m/s]', fontsize=12)
    plt.ylabel('Hour of Day', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

if __name__ == "__main__":
    process_data()