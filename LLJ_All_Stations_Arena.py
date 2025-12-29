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
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/all_stations_arena'
LOG_DIR = r'/home/huxun/02_LLJ/logs'

# 判定标准
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 60      
MAX_JET_HEIGHT = 480
# ===========================================

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'station_fits'), exist_ok=True) # 存放单场站图片的子目录
os.makedirs(LOG_DIR, exist_ok=True)

# --- 0. 配置日志系统 ---
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f'LLJ_All_Stations_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"日志系统初始化完成。日志文件: {log_file}")

# ---------------------------------------------------------
# 1. 定义所有参赛选手 (数学模型)
# ---------------------------------------------------------

def model_banta(z, alpha, beta):
    """ [选手A] Banta (Wall Jet) """
    z = np.maximum(z, 1e-6)
    val = np.power(z, alpha) * np.exp(beta * (1.0 - z))
    return np.nan_to_num(val)

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
    val = 1.0 - k * (z - 1.0)**2
    return np.maximum(val, 0)

# ---------------------------------------------------------
# 2. 极速版数据读取与归一化 (向量化加速)
# ---------------------------------------------------------
def read_and_normalize(file_path):
    station_name = os.path.basename(file_path).split('-')[0]
    # logging.info(f"正在读取: {station_name}")
    
    # 1. 读取文件
    df = None
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16']
    for enc in encodings:
        try:
            temp_df = pd.read_csv(file_path, sep='\t', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp_df.columns):
                df = temp_df
                break
            temp_df = pd.read_csv(file_path, sep='\s+', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp_df.columns):
                df = temp_df
                break
        except: continue
        
    if df is None:
        logging.error(f"{station_name}: 无法读取文件。")
        return None, None
    
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    
    # 2. 矩阵化提取
    speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    if not heights: return None, None

    n_samples = len(df)
    n_heights = len(heights)
    ws_matrix = np.full((n_samples, n_heights), np.nan)
    
    for i, h in enumerate(heights):
        col = [c for c in df.columns if f'{h}m水平风速' in c and '最大' not in c][0]
        ws_matrix[:, i] = pd.to_numeric(df[col], errors='coerce').values
        
    # 3. 向量化计算
    valid_mask = ~np.isnan(ws_matrix).any(axis=1)
    ws_matrix = ws_matrix[valid_mask]
    
    if len(ws_matrix) == 0: return None, None

    max_indices = np.argmax(ws_matrix, axis=1)
    u_jets = ws_matrix[np.arange(len(ws_matrix)), max_indices]
    z_jets = np.array(heights)[max_indices]
    
    u_bottoms = ws_matrix[:, 0]
    u_tops = ws_matrix[:, -1]
    
    # 4. 筛选
    cond_h = (z_jets >= MIN_JET_HEIGHT) & (z_jets <= MAX_JET_HEIGHT)
    cond_shear = (u_jets - u_bottoms >= LLJ_THRESHOLD) & (u_jets - u_tops >= LLJ_THRESHOLD)
    final_mask = cond_h & cond_shear
    
    ws_final = ws_matrix[final_mask]
    u_jets_final = u_jets[final_mask]
    z_jets_final = z_jets[final_mask]
    
    if len(ws_final) < 10: # 样本太少就跳过
        logging.warning(f"{station_name}: 有效样本不足 ({len(ws_final)})")
        return None, None
    
    # 5. 归一化
    norm_u_matrix = ws_final / u_jets_final[:, np.newaxis]
    z_grid = np.array(heights)[np.newaxis, :] 
    norm_z_matrix = z_grid / z_jets_final[:, np.newaxis]
    
    flat_norm_z = norm_z_matrix.flatten()
    flat_norm_u = norm_u_matrix.flatten()
    
    logging.info(f"{station_name}: 提取成功 (N={len(ws_final)})")
    return flat_norm_z, flat_norm_u

# ---------------------------------------------------------
# 3. 单场站模型竞技函数
# ---------------------------------------------------------
def run_station_arena(nz, nu, station_name):
    """在一个场站数据上跑所有模型，返回结果列表"""
    z_fit = np.linspace(0, 2.5, 200)
    models_res = []

    # 定义模型列表以方便循环
    candidates = [
        {'name': 'Banta', 'func': model_banta, 'p0': [1.0, 1.0], 'bounds': ([0,0], [10,10]), 'color': 'red'},
        {'name': 'Gaussian', 'func': model_gaussian, 'p0': [0.5], 'bounds': None, 'color': 'blue'},
        {'name': 'Asym-Gauss', 'func': model_asym_gaussian, 'p0': [0.4, 0.6], 'bounds': None, 'color': 'green'},
        {'name': 'Sech', 'func': model_sech, 'p0': [0.5, 1.0], 'bounds': None, 'color': 'purple'},
        {'name': 'Quadratic', 'func': model_quadratic, 'p0': [0.5], 'bounds': ([0], [10]), 'color': 'orange'}
    ]
    
    # 绘图初始化
    plt.figure(figsize=(10, 8))
    # 降采样绘图背景
    if len(nz) > 5000:
        idx = np.random.choice(len(nz), 5000, replace=False)
        plt.scatter(nu[idx], nz[idx], s=1, color='gray', alpha=0.1, label='Raw Data')
    else:
        plt.scatter(nu, nz, s=1, color='gray', alpha=0.1, label='Raw Data')

    best_rmse = float('inf')
    best_model_name = ""

    for model in candidates:
        try:
            kwargs = {'p0': model['p0'], 'maxfev': 2000}
            if model['bounds']: kwargs['bounds'] = model['bounds']
            
            popt, _ = curve_fit(model['func'], nz, nu, **kwargs)
            rmse = np.sqrt(np.mean((nu - model['func'](nz, *popt))**2))
            
            # 记录结果
            res_entry = {
                'Model': model['name'],
                'RMSE': rmse,
                'Params': str(np.round(popt, 3))
            }
            models_res.append(res_entry)
            
            # 更新最佳
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model['name']

            # 画线
            label_txt = f"{model['name']} (RMSE={rmse:.3f})"
            lw = 3 if model['name'] == best_model_name else 1.5 # 最佳模型加粗暂定，后面会覆盖，这里主要画全
            plt.plot(model['func'](z_fit, *popt), z_fit, color=model['color'], lw=2, label=label_txt)

        except:
            continue

    # 完善绘图
    plt.title(f'Station: {station_name} | Winner: {best_model_name}', fontsize=14)
    plt.xlabel('Normalized Speed')
    plt.ylabel('Normalized Height')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 2.5)
    
    # 保存图片
    img_name = f"{station_name}_best_fit.png"
    plt.savefig(os.path.join(OUTPUT_DIR, 'station_fits', img_name), dpi=150)
    plt.close() # 关闭画布释放内存

    # 返回按 RMSE 排序的结果
    models_res.sort(key=lambda x: x['RMSE'])
    return models_res

# ---------------------------------------------------------
# 4. 主程序
# ---------------------------------------------------------
def main():
    setup_logging()
    
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    if not files:
        logging.error(f"目录 {DATA_DIR} 下未找到数据文件！")
        return
    
    logging.info(f"检测到 {len(files)} 个文件，开始批量模型竞技...")
    
    all_station_summary = [] # 存储每个场站的冠军信息
    winner_counts = {}       # 统计各模型夺冠次数

    for idx, f in enumerate(files):
        station_name = os.path.basename(f).split('-')[0]
        # logging.info(f"--- 处理 [{idx+1}/{len(files)}]: {station_name} ---")
        
        nz, nu = read_and_normalize(f)
        if nz is None: continue
        
        # 跑竞技场
        results = run_station_arena(nz, nu, station_name)
        
        if results:
            winner = results[0] # RMSE 最小的
            second = results[1] if len(results) > 1 else None
            
            # 记录冠军
            summary_entry = {
                'Station': station_name,
                'Best_Model': winner['Model'],
                'Best_RMSE': winner['RMSE'],
                'Best_Params': winner['Params'],
                'Second_Model': second['Model'] if second else 'None',
                'Second_RMSE': second['RMSE'] if second else 0,
                'RMSE_Improvement': 0
            }
            
            if second:
                imp = (second['RMSE'] - winner['RMSE']) / second['RMSE'] * 100
                summary_entry['RMSE_Improvement'] = round(imp, 2)
                
            all_station_summary.append(summary_entry)
            
            # 统计
            w_name = winner['Model']
            winner_counts[w_name] = winner_counts.get(w_name, 0) + 1
            
            logging.info(f"  -> 冠军: {w_name} (RMSE={winner['RMSE']:.4f})")
        else:
            logging.warning(f"  -> {station_name} 拟合全部失败")

    # --- 5. 生成最终报告 ---
    if not all_station_summary:
        logging.error("没有产生任何有效结果。")
        return

    df_res = pd.DataFrame(all_station_summary)
    
    # 排序：按场站名
    df_res = df_res.sort_values('Station')
    
    # 保存 Excel
    out_excel = os.path.join(OUTPUT_DIR, 'All_Stations_Best_Models.xlsx')
    df_res.to_excel(out_excel, index=False)
    
    # 打印总结
    print("\n" + "="*50)
    print(" 🏆 全场站模型竞技总决赛结果 🏆")
    print("="*50)
    print(f"处理场站数: {len(all_station_summary)}")
    print("\n[夺冠榜单]")
    sorted_counts = sorted(winner_counts.items(), key=lambda x: x[1], reverse=True)
    for model, count in sorted_counts:
        print(f"  - {model:12s}: 夺冠 {count} 次")
    
    print("\n[详细结果已保存]")
    print(f"  - Excel 总表: {out_excel}")
    print(f"  - 单站拟合图: {os.path.join(OUTPUT_DIR, 'station_fits')}")
    print("="*50)
    
    logging.info("任务全部完成。")

if __name__ == "__main__":
    main()