import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
import glob
import logging
from datetime import datetime

# ================= 配置区域 =================
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/model_arena'
LOG_DIR = r'/home/huxun/02_LLJ/logs'

# 判定标准
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 60      # <--- 已调整为 60m
MAX_JET_HEIGHT = 480
# ===========================================

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- 0. 配置日志系统 ---
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f'LLJ_Arena_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler() # 同时输出到控制台
        ]
    )
    logging.info(f"日志系统初始化完成。日志文件: {log_file}")
    logging.info(f"配置参数: Min_H={MIN_JET_HEIGHT}m, Max_H={MAX_JET_HEIGHT}m, Threshold={LLJ_THRESHOLD}m/s")

# ---------------------------------------------------------
# 1. 定义所有参赛选手 (数学模型)
# ---------------------------------------------------------

def model_banta(z, alpha, beta):
    """ [选手A] Banta (Wall Jet) """
    # 为防止计算溢出，做一点保护
    z = np.maximum(z, 1e-6) # 避免 0 的幂运算报错
    val = np.power(z, alpha) * np.exp(beta * (1.0 - z))
    return np.nan_to_num(val) # 处理可能出现的无穷大

def model_gaussian(z, sigma):
    """ [选手B] 标准高斯 (Symmetric Gaussian) """
    return np.exp(-((z - 1.0)**2) / (2 * sigma**2))

def model_asym_gaussian(z, sigma_down, sigma_up):
    """ [选手C] 非对称高斯 (Asymmetric Gaussian) """
    sigma = np.where(z <= 1.0, sigma_down, sigma_up)
    return np.exp(-((z - 1.0)**2) / (2 * sigma**2))

def model_sech(z, width, shape):
    """ [选手D] 双曲正割 (Sech / Modified Tanh 变体) """
    return (1.0 / np.cosh((z - 1.0) / width)) ** shape

def model_quadratic(z, k):
    """ [选手E] 二次函数 (倒扣抛物线) """
    # 模拟急流核心附近的对称变化：U = 1 - k*(z-1)^2
    # 限制不小于0，避免出现负风速
    val = 1.0 - k * (z - 1.0)**2
    return np.maximum(val, 0)

# ---------------------------------------------------------
# 2. 极速版数据读取与归一化 (向量化加速)
# ---------------------------------------------------------
def read_and_normalize(file_path):
    logging.info(f"正在读取文件 (极速模式): {os.path.basename(file_path)}")
    
    # 1. 读取文件
    df = None
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16']
    for enc in encodings:
        try:
            # 尝试跳过前12行读取
            temp_df = pd.read_csv(file_path, sep='\t', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp_df.columns):
                df = temp_df
                break
            # 尝试不定长空格
            temp_df = pd.read_csv(file_path, sep='\s+', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp_df.columns):
                df = temp_df
                break
        except: continue
        
    if df is None:
        logging.error("无法读取文件或找不到表头。")
        return None, None
    
    # 清洗列名
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    
    # 2. 矩阵化提取
    # 找到所有风速列
    speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    logging.info(f"识别到高度层: {heights}")
    
    if not heights: return None, None

    # 构建大矩阵 [N_samples, N_heights]
    # 使用 NumPy 快速构建，避免行循环
    n_samples = len(df)
    n_heights = len(heights)
    ws_matrix = np.full((n_samples, n_heights), np.nan)
    
    for i, h in enumerate(heights):
        col = [c for c in df.columns if f'{h}m水平风速' in c and '最大' not in c][0]
        # 强制转数字，非数字变NaN
        ws_matrix[:, i] = pd.to_numeric(df[col], errors='coerce').values
        
    # 3. 向量化计算急流
    # 排除全NaN行
    valid_mask = ~np.isnan(ws_matrix).any(axis=1)
    ws_matrix = ws_matrix[valid_mask]
    
    if len(ws_matrix) == 0: 
        logging.warning("无有效数据行。")
        return None, None

    # 找最大值索引和值
    max_indices = np.argmax(ws_matrix, axis=1) # 每个时刻最大风速的列索引
    u_jets = ws_matrix[np.arange(len(ws_matrix)), max_indices] # 取出最大风速值
    z_jets = np.array(heights)[max_indices] # 取出对应的物理高度
    
    # 提取底层和顶层风速
    u_bottoms = ws_matrix[:, 0]
    u_tops = ws_matrix[:, -1]
    
    # 4. 向量化筛选 (Boolean Indexing)
    # 条件1: 高度范围 (60m - 480m)
    cond_h = (z_jets >= MIN_JET_HEIGHT) & (z_jets <= MAX_JET_HEIGHT)
    # 条件2: 切变强度 (> 2.0 m/s)
    cond_shear = (u_jets - u_bottoms >= LLJ_THRESHOLD) & (u_jets - u_tops >= LLJ_THRESHOLD)
    
    final_mask = cond_h & cond_shear
    
    # 应用筛选
    ws_final = ws_matrix[final_mask]
    u_jets_final = u_jets[final_mask]
    z_jets_final = z_jets[final_mask]
    
    if len(ws_final) == 0: 
        logging.warning("未筛选出符合条件的急流样本。")
        return None, None
    
    # 5. 向量化归一化
    # 广播机制：矩阵除以向量
    norm_u_matrix = ws_final / u_jets_final[:, np.newaxis]
    
    # 高度归一化
    # 构造高度网格 [1, N_heights]
    z_grid = np.array(heights)[np.newaxis, :] 
    # 归一化高度矩阵 [N_samples, N_heights]
    norm_z_matrix = z_grid / z_jets_final[:, np.newaxis]
    
    # 展平为一维数组，供 curve_fit 使用
    flat_norm_z = norm_z_matrix.flatten()
    flat_norm_u = norm_u_matrix.flatten()
    
    logging.info(f"提取完成: 有效样本 {len(ws_final)} 个，归一化数据点 {len(flat_norm_z)} 个")
    return flat_norm_z, flat_norm_u

# ---------------------------------------------------------
# 3. 竞技场主逻辑
# ---------------------------------------------------------
def main():
    setup_logging()
    
    # 自动寻找第一个 txt 文件
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    if not files:
        logging.error(f"目录 {DATA_DIR} 下未找到数据文件！")
        return
    
    # 默认取第一个文件，或者你可以遍历所有文件把数据加在一起
    # 这里演示单文件分析，如果你想跑所有文件，可以在这里加个循环把 nz, nu extend 起来
    target_file = files[0] 
    logging.info(f"选中测试文件: {target_file}")

    nz, nu = read_and_normalize(target_file)
    
    if nz is None or len(nz) < 100:
        logging.warning("数据不足，终止分析。")
        return
    
    # 准备绘图
    plt.figure(figsize=(12, 9))
    # 画一部分散点作为背景 (如果点太多，只画 10000 个)
    if len(nz) > 10000:
        idx = np.random.choice(len(nz), 10000, replace=False)
        plt.scatter(nu[idx], nz[idx], s=1, color='gray', alpha=0.1, label='Raw Data (Sampled)')
    else:
        plt.scatter(nu, nz, s=1, color='gray', alpha=0.1, label='Raw Data')
    
    z_fit = np.linspace(0, 2.5, 200)
    results = []

    logging.info(">>> 开始模型比武 (Model Arena) <<<")

    # --- 1. Banta ---
    try:
        # 添加 bounds 防止参数跑飞
        popt, _ = curve_fit(model_banta, nz, nu, p0=[1.0, 1.0], bounds=([0, 0], [10, 10]), maxfev=2000)
        rmse = np.sqrt(np.mean((nu - model_banta(nz, *popt))**2))
        plt.plot(model_banta(z_fit, *popt), z_fit, 'r-', lw=2, label=f'Banta (RMSE={rmse:.4f})')
        results.append(('Banta', rmse))
        logging.info(f"Model [Banta] finished. RMSE={rmse:.4f}, Params={popt}")
    except Exception as e: logging.error(f"Banta failed: {e}")

    # --- 2. Gaussian ---
    try:
        popt, _ = curve_fit(model_gaussian, nz, nu, p0=[0.5], maxfev=2000)
        rmse = np.sqrt(np.mean((nu - model_gaussian(nz, *popt))**2))
        plt.plot(model_gaussian(z_fit, *popt), z_fit, 'b--', lw=2, label=f'Gaussian (RMSE={rmse:.4f})')
        results.append(('Gaussian', rmse))
        logging.info(f"Model [Gaussian] finished. RMSE={rmse:.4f}, Params={popt}")
    except Exception as e: logging.error(f"Gaussian failed: {e}")

    # --- 3. Asym Gaussian ---
    try:
        popt, _ = curve_fit(model_asym_gaussian, nz, nu, p0=[0.4, 0.6], maxfev=2000)
        rmse = np.sqrt(np.mean((nu - model_asym_gaussian(nz, *popt))**2))
        plt.plot(model_asym_gaussian(z_fit, *popt), z_fit, 'g-.', lw=2, label=f'Asym-Gauss (RMSE={rmse:.4f})')
        results.append(('Asym-Gauss', rmse))
        logging.info(f"Model [Asym-Gauss] finished. RMSE={rmse:.4f}, Params={popt}")
    except Exception as e: logging.error(f"Asym-Gauss failed: {e}")
    
    # --- 4. Sech ---
    try:
        popt, _ = curve_fit(model_sech, nz, nu, p0=[0.5, 1.0], maxfev=2000)
        rmse = np.sqrt(np.mean((nu - model_sech(nz, *popt))**2))
        plt.plot(model_sech(z_fit, *popt), z_fit, 'm:', lw=2, label=f'Sech (RMSE={rmse:.4f})')
        results.append(('Sech', rmse))
        logging.info(f"Model [Sech] finished. RMSE={rmse:.4f}, Params={popt}")
    except Exception as e: logging.error(f"Sech failed: {e}")

    # --- 5. Quadratic (新增) ---
    try:
        popt, _ = curve_fit(model_quadratic, nz, nu, p0=[0.5], bounds=([0], [10]), maxfev=2000)
        rmse = np.sqrt(np.mean((nu - model_quadratic(nz, *popt))**2))
        plt.plot(model_quadratic(z_fit, *popt), z_fit, color='orange', linestyle='-', lw=3, label=f'Quadratic (RMSE={rmse:.4f})')
        results.append(('Quadratic', rmse))
        logging.info(f"Model [Quadratic] finished. RMSE={rmse:.4f}, Params={popt}")
    except Exception as e: logging.error(f"Quadratic failed: {e}")

    # --- 结算 ---
    plt.title(f'Model Arena (MinHeight={MIN_JET_HEIGHT}m)\nComparison of 5 Models', fontsize=14)
    plt.xlabel('Normalized Speed (U/U_jet)')
    plt.ylabel('Normalized Height (Z/Z_jet)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 2.5)
    
    img_path = os.path.join(OUTPUT_DIR, 'Model_Comparison_Final.png')
    plt.savefig(img_path, dpi=300)
    logging.info(f"对比图已保存: {img_path}")
    
    # 打印排名
    results.sort(key=lambda x: x[1])
    logging.info("-" * 40)
    logging.info(" 🏆 最终排名 (RMSE 越小越好)")
    logging.info("-" * 40)
    print("\n" + "="*40)
    print(" 🏆 比赛结果 ")
    print("="*40)
    for rank, (name, err) in enumerate(results):
        res_str = f" {rank+1}. {name:12s} | RMSE: {err:.5f}"
        print(res_str)
        logging.info(res_str)

if __name__ == "__main__":
    main()