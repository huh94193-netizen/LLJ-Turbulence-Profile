import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
import glob
import logging
from datetime import datetime
import warnings

# ================= 配置区域 =================
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/wd_all_stations'
LOG_DIR = r'/home/huxun/02_LLJ/logs'

# 判定标准
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'station_wd_fits'), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 0. 配置日志 ---
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f'WD_All_Stations_{timestamp}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
    logging.info(f"全场站风向廓线拟合任务启动。")

# --- 1. 定义相对模型 ---
# 注意：输入是 delta_z (z - z_base)，输出是 delta_wd (wd - wd_base)
# 截距强制为 0，因为在 z_base 处偏转必然为 0

def model_linear_relative(delta_z, rate):
    """线性偏转: dWD = rate * dZ"""
    return rate * delta_z

def model_quadratic_relative(delta_z, rate, curve):
    """二次偏转: dWD = rate * dZ + curve * dZ^2"""
    return rate * delta_z + curve * delta_z**2

# --- 2. 辅助工具 ---
def unwrap_deg(degrees):
    """解缠绕：350->10 变为 350->370"""
    rads = np.radians(degrees)
    unwrapped = np.unwrap(rads)
    return np.degrees(unwrapped)

def strict_tab_parse_v3(file_path):
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16', 'latin-1']
    for enc in encodings:
        try:
            # 尝试跳过前12行读取
            temp = pd.read_csv(file_path, sep='\t', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp.columns): return temp
            temp = pd.read_csv(file_path, sep='\s+', skiprows=12, encoding=enc, engine='python')
            if 'm水平风速' in str(temp.columns): return temp
        except: continue
    return None

# --- 3. 单场站处理函数 ---
def analyze_single_station(file_path):
    station_name = os.path.basename(file_path).split('-')[0]
    # logging.info(f"正在分析: {station_name}")
    
    df = strict_tab_parse_v3(file_path)
    if df is None: return None

    # 清洗列名
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    
    # 映射列名
    speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    ws_cols_map = {}
    wd_cols_map = {}
    for h in heights:
        ws_c = [c for c in df.columns if f'{h}m水平风速' in c and '最大' not in c][0]
        wd_c = next((c for c in df.columns if str(h) in c and ('风向' in c or 'Direction' in c) and '最大' not in c), None)
        if ws_c and wd_c:
            ws_cols_map[h] = ws_c
            wd_cols_map[h] = wd_c

    # 提取急流事件的相对风向数据
    all_dz = []
    all_dwd = []
    event_count = 0
    
    for idx in df.index:
        # 提取风速判 LLJ
        ws_vals = []
        valid_h = []
        for h in heights:
            if h in ws_cols_map:
                try:
                    v = float(df.loc[idx, ws_cols_map[h]])
                    ws_vals.append(v)
                    valid_h.append(h)
                except: pass
        
        if not ws_vals: continue
        ws_arr = np.array(ws_vals)
        mx_i = np.argmax(ws_arr)
        
        # LLJ 判定
        if (valid_h[mx_i] > MIN_JET_HEIGHT) and (valid_h[mx_i] < MAX_JET_HEIGHT):
            if (ws_arr[mx_i] - ws_arr[0] >= LLJ_THRESHOLD) and (ws_arr[mx_i] - ws_arr[-1] >= LLJ_THRESHOLD):
                # 是急流，提取风向
                wd_vals = []
                final_h = []
                for h in valid_h:
                    if h in wd_cols_map:
                        try:
                            d = float(df.loc[idx, wd_cols_map[h]])
                            if not np.isnan(d):
                                wd_vals.append(d)
                                final_h.append(h)
                        except: pass
                
                if len(wd_vals) > 4:
                    z = np.array(final_h)
                    wd = np.array(wd_vals)
                    # 相对化处理
                    wd_cont = unwrap_deg(wd)
                    wd_base = wd_cont[0]
                    z_base = z[0]
                    
                    all_dz.extend(z - z_base)
                    all_dwd.extend(wd_cont - wd_base)
                    event_count += 1

    if event_count < 10:
        logging.warning(f"{station_name}: 急流样本不足 ({event_count})")
        return None

    # --- 模型竞技 ---
    all_dz = np.array(all_dz)
    all_dwd = np.array(all_dwd)
    
    # 过滤极端值 (偏转 > 120度视为异常)
    mask = np.abs(all_dwd) < 120
    dz_fit = all_dz[mask]
    dwd_fit = all_dwd[mask]
    
    # 1. Linear Fit
    rmse_lin = np.nan
    popt_lin = [np.nan]
    try:
        popt_lin, _ = curve_fit(model_linear_relative, dz_fit, dwd_fit)
        pred = model_linear_relative(dz_fit, *popt_lin)
        rmse_lin = np.sqrt(np.mean((dwd_fit - pred)**2))
    except: pass
    
    # 2. Quadratic Fit
    rmse_quad = np.nan
    popt_quad = [np.nan, np.nan]
    try:
        popt_quad, _ = curve_fit(model_quadratic_relative, dz_fit, dwd_fit)
        pred = model_quadratic_relative(dz_fit, *popt_quad)
        rmse_quad = np.sqrt(np.mean((dwd_fit - pred)**2))
    except: pass

    # 判定胜者
    best_model = "None"
    if not np.isnan(rmse_lin) and not np.isnan(rmse_quad):
        # 如果 Quadratic 提升很小 (<2%)，优先选 Linear (奥卡姆剃刀)
        if (rmse_lin - rmse_quad) / rmse_lin < 0.02:
            best_model = "Linear"
        else:
            best_model = "Quadratic"
    elif not np.isnan(rmse_lin):
        best_model = "Linear"
    
    # --- 绘图 ---
    plt.figure(figsize=(8, 6))
    # 降采样散点
    if len(dz_fit) > 3000:
        idx_samp = np.random.choice(len(dz_fit), 3000, replace=False)
        plt.scatter(dwd_fit[idx_samp], dz_fit[idx_samp], c='gray', s=1, alpha=0.2, label='Relative WD Samples')
    else:
        plt.scatter(dwd_fit, dz_fit, c='gray', s=1, alpha=0.4, label='Relative WD Samples')
        
    z_smooth = np.linspace(0, np.max(dz_fit), 100)
    
    if not np.isnan(rmse_lin):
        y_lin = model_linear_relative(z_smooth, *popt_lin)
        lw = 3 if best_model == "Linear" else 1.5
        alpha = 1.0 if best_model == "Linear" else 0.6
        plt.plot(y_lin, z_smooth, 'b-', lw=lw, alpha=alpha, label=f'Linear (RMSE={rmse_lin:.2f})')
        
    if not np.isnan(rmse_quad):
        y_quad = model_quadratic_relative(z_smooth, *popt_quad)
        lw = 3 if best_model == "Quadratic" else 1.5
        alpha = 1.0 if best_model == "Quadratic" else 0.6
        plt.plot(y_quad, z_smooth, 'r--', lw=lw, alpha=alpha, label=f'Quadratic (RMSE={rmse_quad:.2f})')
        
    plt.title(f'{station_name} WD Profile | Winner: {best_model}')
    plt.xlabel('Relative Veering [deg]')
    plt.ylabel('Relative Height [m]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    img_name = f"{station_name}_WD_Fit.png"
    plt.savefig(os.path.join(OUTPUT_DIR, 'station_wd_fits', img_name), dpi=150)
    plt.close()
    
    logging.info(f"  -> {station_name}: Best={best_model}, LinRMSE={rmse_lin:.3f}, QuadRMSE={rmse_quad:.3f}")

    return {
        'Station': station_name,
        'Events': event_count,
        'Best_Model': best_model,
        'Linear_Rate': popt_lin[0],
        'Quad_Rate': popt_quad[0] if len(popt_quad)>1 else np.nan,
        'Quad_Curve': popt_quad[1] if len(popt_quad)>1 else np.nan,
        'RMSE_Linear': rmse_lin,
        'RMSE_Quad': rmse_quad
    }

# --- 4. 主程序 ---
def main():
    setup_logging()
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    
    results = []
    
    for f in files:
        res = analyze_single_station(f)
        if res:
            results.append(res)
            
    if not results:
        logging.error("未生成任何结果")
        return
        
    # 保存 Excel
    df_res = pd.DataFrame(results)
    # 重新排序列
    cols = ['Station', 'Events', 'Best_Model', 'Linear_Rate', 'Quad_Rate', 'Quad_Curve', 'RMSE_Linear', 'RMSE_Quad']
    df_res = df_res[cols]
    
    out_excel = os.path.join(OUTPUT_DIR, 'WD_All_Stations_Summary.xlsx')
    df_res.to_excel(out_excel, index=False)
    
    print("="*60)
    print(" 全场站风向廓线拟合完成 ")
    print("="*60)
    print(f"1. 汇总表格: {out_excel}")
    print(f"2. 单站图片: {os.path.join(OUTPUT_DIR, 'station_wd_fits')}")
    print("\n[统计摘要]")
    print(df_res['Best_Model'].value_counts())
    print("\n[平均线性偏转率]")
    mean_rate = df_res['Linear_Rate'].mean()
    print(f"  {mean_rate:.4f} deg/m (即每上升100m，风向右偏约 {mean_rate*100:.1f} 度)")

if __name__ == "__main__":
    main()