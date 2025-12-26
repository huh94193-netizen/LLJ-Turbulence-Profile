import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
from sklearn.metrics import r2_score, mean_squared_error

# ================= 配置 =================
# 选一个数据量大的场站文件进行测试
FILE_PATH = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/model_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 质量控制参数
MIN_AVAILABILITY = 80.0
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# =======================================

# --- 1. 定义候选模型 ---

def model_gaussian(z, u_base, u_jet, z_jet, sigma):
    """
    1. 标准高斯模型 (4参数)
    假设急流上下对称
    """
    return u_base + u_jet * np.exp(-((z - z_jet)**2) / (2 * sigma**2))

def model_asymmetric_gaussian(z, u_base, u_jet, z_jet, sigma_down, sigma_up):
    """
    2. 非对称高斯模型 (5参数) - 推荐
    允许急流核心下方(down)和上方(up)的衰减速率不同
    通常下方受地面摩擦影响，sigma_down 会更小(切变更大)
    """
    sigma = np.where(z <= z_jet, sigma_down, sigma_up)
    return u_base + u_jet * np.exp(-((z - z_jet)**2) / (2 * sigma**2))

def model_tanh_log(z, u_star, z0, u_mix, z_inv, width):
    """
    3. 对数律 + Tanh (5参数)
    用于模拟混合层顶部的急转弯
    注意：这个公式对初值非常敏感，容易不收敛
    """
    # 简化版 Tanh 过渡模型
    # 这是一个示意性公式，实际 Tanh 模型非常复杂
    return (u_star / 0.4) * np.log(z / z0 + 1) + (u_mix / 2) * (1 + np.tanh((z - z_inv) / width))

# --- 2. 核心处理逻辑 ---

def process_and_compare(file_path):
    print(f"正在分析文件: {os.path.basename(file_path)}")
    
    # A. 解析数据 (复用之前逻辑)
    # ... (此处省略重复的解析代码，直接使用简化版读取) ...
    # 为了代码简洁，这里假设你已经有办法读入 df_raw
    # 实际运行时请把之前的 strict_tab_parse_v3 贴进来
    df_raw = strict_tab_parse_v3(file_path) 
    if df_raw is None: return

    # 清洗列名
    df_raw.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df_raw.columns]
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c and '最小' not in c and '偏差' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    # 提取数值
    data_dict = {}
    for h in heights:
        data_dict[f'ws_{h}'] = pd.to_numeric(df_raw[f'{h}m水平风速'], errors='coerce')
    df_calc = pd.DataFrame(data_dict)
    
    # 筛选急流
    ws_cols = [f'ws_{h}' for h in heights]
    df_clean = df_calc.dropna(subset=ws_cols)
    speed_mat = df_clean[ws_cols].values
    
    llj_events = [] # 存储每个事件的 (heights, speeds)
    
    for i in range(len(speed_mat)):
        s = speed_mat[i]
        try:
            mx_i = np.nanargmax(s)
            if heights[mx_i] <= MIN_JET_HEIGHT or heights[mx_i] >= MAX_JET_HEIGHT: continue
            if (s[mx_i] - s[0] >= LLJ_THRESHOLD) and (s[mx_i] - s[-1] >= LLJ_THRESHOLD):
                llj_events.append(s)
        except: continue
        
    print(f" -> 共捕获 {len(llj_events)} 个 LLJ 事件，开始模型竞技...")
    
    # B. 模型竞技场
    results = {
        'Gaussian': {'rmse': [], 'success': 0},
        'Asym_Gaussian': {'rmse': [], 'success': 0},
        # 'Tanh_Log': {'rmse': [], 'success': 0} # Tanh太难收敛，先注释掉，以免报错太多
    }
    
    z_vals = np.array(heights)
    
    for i, u_vals in enumerate(llj_events):
        if i % 100 == 0: print(f"    处理进度: {i}/{len(llj_events)}")
        
        # 初值猜测 (重要)
        max_u = np.max(u_vals)
        max_z = z_vals[np.argmax(u_vals)]
        
        # --- 1. Gaussian ---
        try:
            # p0: [u_base, u_jet, z_jet, sigma]
            p0_g = [u_vals[0], max_u - u_vals[0], max_z, 50]
            popt, _ = curve_fit(model_gaussian, z_vals, u_vals, p0=p0_g, maxfev=2000)
            
            # 计算 RMSE
            u_pred = model_gaussian(z_vals, *popt)
            rmse = np.sqrt(mean_squared_error(u_vals, u_pred))
            
            results['Gaussian']['rmse'].append(rmse)
            results['Gaussian']['success'] += 1
        except: pass
        
        # --- 2. Asymmetric Gaussian ---
        try:
            # p0: [u_base, u_jet, z_jet, sigma_down, sigma_up]
            p0_ag = [u_vals[0], max_u - u_vals[0], max_z, 40, 60]
            # bounds: sigma > 0
            popt, _ = curve_fit(model_asymmetric_gaussian, z_vals, u_vals, p0=p0_ag, maxfev=2000)
            
            u_pred = model_asymmetric_gaussian(z_vals, *popt)
            rmse = np.sqrt(mean_squared_error(u_vals, u_pred))
            
            results['Asym_Gaussian']['rmse'].append(rmse)
            results['Asym_Gaussian']['success'] += 1
        except: pass

    # C. 统计与绘图
    print("\n" + "="*40)
    print("模型评估报告")
    print("="*40)
    
    plot_data = []
    labels = []
    
    for name, res in results.items():
        n_total = len(llj_events)
        n_success = res['success']
        rmse_list = np.array(res['rmse'])
        
        if len(rmse_list) == 0:
            print(f"{name}: 全部失败")
            continue
            
        mean_rmse = np.mean(rmse_list)
        median_rmse = np.median(rmse_list)
        success_rate = (n_success / n_total) * 100
        
        print(f"[{name}]")
        print(f"  - 拟合成功率: {success_rate:.1f}% ({n_success}/{n_total})")
        print(f"  - 平均 RMSE: {mean_rmse:.3f} m/s")
        print(f"  - 中位 RMSE: {median_rmse:.3f} m/s (推荐参考)")
        
        plot_data.append(rmse_list)
        labels.append(f"{name}\n(Rate: {success_rate:.0f}%)")

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot(plot_data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="lightblue"), 
                medianprops=dict(color="red", linewidth=2))
    plt.title(f'Model Error Distribution (RMSE) - N={len(llj_events)}', fontsize=14)
    plt.ylabel('RMSE [m/s] (Lower is Better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    # 去掉极端的离群点显示，让图好看点
    plt.ylim(0, 2.0) 
    
    out_img = os.path.join(OUTPUT_DIR, 'Model_Comparison_Boxplot.png')
    plt.savefig(out_img, dpi=300)
    print(f"\n[图表] 误差分布对比图已保存: {out_img}")

# 辅助函数 (必须包含，否则跑不起来)
def strict_tab_parse_v3(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'latin-1']
    raw_lines = []
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                raw_lines = f.readlines()
            break
        except: continue
    if not raw_lines: return None
    header_idx = -1
    for i, line in enumerate(raw_lines[:100]):
        if "Date/Time" in line or "m水平风速" in line:
            header_idx = i
            break     
    if header_idx == -1: return None
    header = raw_lines[header_idx].strip().split('\t')
    header = [h.strip().replace('"', '') for h in header]
    data = []
    for i in range(header_idx + 1, len(raw_lines)):
        line = raw_lines[i].strip()
        if not line: continue
        parts = line.split('\t')
        parts = [p.strip().replace('"', '') for p in parts]
        if len(parts) > len(header): parts = parts[:len(header)]
        elif len(parts) < len(header): parts += [''] * (len(header) - len(parts))
        data.append(parts)
    return pd.DataFrame(data, columns=header)

if __name__ == "__main__":
    process_and_compare(FILE_PATH)