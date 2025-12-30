import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.font_manager as fm
import re
import os
import glob
import warnings

# ================= 配置区域 =================
INPUT_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/ti_arena_all_stations'

# 质量控制
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --- 0. 字体配置 ---
def configure_chinese_font():
    font_candidates = ['WenQuanYi Micro Hei', 'Zen Hei', 'Droid Sans Fallback', 'SimHei', 'Microsoft YaHei', 'SimSun']
    for font_name in font_candidates:
        try:
            if fm.findfont(font_name) != fm.findfont('DejaVu Sans'):
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                return font_name
        except: continue
    return None

sys_font = configure_chinese_font()

# --- 1. 定义四大模型群 (No Tanh, Include Sech) ---

def model_inv_gaussian(z, ti_base, ti_dip, z_jet, sigma):
    """ [模型 1] 倒置高斯 (基础款) """
    return ti_base - ti_dip * np.exp(-((z - z_jet)**2) / (2 * sigma**2))

def model_inv_asym_gaussian(z, ti_base, ti_dip, z_jet, sigma_down, sigma_up):
    """ [模型 2] 倒置非对称高斯 (进阶款) """
    sigma = np.where(z <= z_jet, sigma_down, sigma_up)
    return ti_base - ti_dip * np.exp(-((z - z_jet)**2) / (2 * sigma**2))

def model_inv_banta(z, ti_base, ti_dip, z_jet, alpha, beta):
    """ [模型 3] 倒置 Banta (形状灵活) """
    z_norm = z / z_jet
    z_norm = np.maximum(z_norm, 1e-6)
    shape = np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))
    return ti_base - ti_dip * shape

def model_inv_sech(z, ti_base, ti_dip, z_jet, width):
    """ [模型 4] 倒置 Sech (深井/尖峰) """
    arg = (z - z_jet) / width
    # 保护防止溢出
    val = np.zeros_like(arg)
    mask = np.abs(arg) < 700 
    val[mask] = 1.0 / np.cosh(arg[mask])
    return ti_base - ti_dip * val

# 模型字典配置
MODELS = {
    'Inv_Gaussian': {'func': model_inv_gaussian, 'c': 'blue', 'ls': '--'},
    'Inv_Asym_Gauss': {'func': model_inv_asym_gaussian, 'c': 'green', 'ls': '-.'},
    'Inv_Banta': {'func': model_inv_banta, 'c': 'purple', 'ls': ':'},
    'Inv_Sech': {'func': model_inv_sech, 'c': 'red', 'ls': '-'} # Sech 用醒目的实线
}

# --- 2. 数据处理函数 ---
def strict_tab_parse_v3(file_path):
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16', 'latin-1']
    header_row = -1
    encoding_used = None
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                lines = [f.readline() for _ in range(50)]
            for i, line in enumerate(lines):
                if "m水平风速" in line and "Date" in line:
                    header_row = i
                    encoding_used = enc
                    break
            if header_row != -1: break
        except: continue

    if header_row == -1: return None

    try:
        df = pd.read_csv(file_path, skiprows=header_row, sep='\t', encoding=encoding_used, engine='python')
    except:
        try:
            df = pd.read_csv(file_path, skiprows=header_row, sep='\s+', encoding=encoding_used, engine='python')
        except: return None
    
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    return df

def get_mean_ti_profile(file_path):
    df_raw = strict_tab_parse_v3(file_path)
    if df_raw is None: return None, None

    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    n = len(df_raw)
    ws_mat = np.full((n, len(heights)), np.nan)
    ti_mat = np.full((n, len(heights)), np.nan)
    
    for i, h in enumerate(heights):
        ws_c = [c for c in df_raw.columns if f'{h}m水平风速' in c and '最大' not in c][0]
        std_c_list = [c for c in df_raw.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c]
        if std_c_list:
            w = pd.to_numeric(df_raw[ws_c], errors='coerce').values
            s = pd.to_numeric(df_raw[std_c_list[0]], errors='coerce').values
            ws_mat[:, i] = w
            with np.errstate(divide='ignore', invalid='ignore'):
                ti = s / w
                ti[w < 3.0] = np.nan
            ti_mat[:, i] = ti
            
    llj_ti_list = []
    for i in range(n):
        u = ws_mat[i, :]
        if np.isnan(u).any(): continue
        mx_i = np.argmax(u)
        z_jet = heights[mx_i]
        
        if (z_jet > MIN_JET_HEIGHT) and (z_jet < MAX_JET_HEIGHT):
            if (u[mx_i] - u[0] >= LLJ_THRESHOLD) and (u[mx_i] - u[-1] >= LLJ_THRESHOLD):
                if not np.isnan(ti_mat[i, :]).any():
                    llj_ti_list.append(ti_mat[i, :])
                    
    if len(llj_ti_list) < 10: return None, None
    return np.array(heights), np.mean(llj_ti_list, axis=0)

# --- 3. 批量竞技场主逻辑 ---
def main():
    print("="*60)
    print(" 全场站多模型拟合大比武 (含 Sech)")
    print("="*60)
    
    files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))
    
    # 结果容器
    summary_results = []
    
    # 绘图准备：10个场站拼图 (3x4)
    n_files = len(files)
    rows = int(np.ceil(n_files / 4))
    fig_all, axes_all = plt.subplots(rows, 4, figsize=(24, 6 * rows))
    axes_flat = axes_all.flatten()
    
    z_smooth = np.linspace(0, 500, 200)

    for idx, f in enumerate(files):
        filename = os.path.basename(f)
        try:
            site_name = filename.split('-')[0]
        except: site_name = filename[:10]
        
        print(f"[{idx+1}/{n_files}] 处理: {site_name} ...", end='')
        
        z_vals, ti_vals = get_mean_ti_profile(f)
        ax = axes_flat[idx]
        
        if z_vals is None:
            print(" 跳过 (数据不足)")
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue
            
        print(" 数据就绪 -> 拟合中...", end='')
        
        # 记录该场站的拟合情况
        station_res = {'Station': site_name, 'Sample_Count': len(ti_vals)} # 这里存样本数不太准，暂略
        
        # 初值准备
        ti_min = np.min(ti_vals)
        ti_max = np.max(ti_vals)
        z_at_min = z_vals[np.argmin(ti_vals)]
        
        best_rmse = 999
        best_model_name = ""
        
        # --- 循环跑 4 个模型 ---
        # 绘图原始点
        ax.plot(ti_vals, z_vals, 'ko', markersize=5, alpha=0.5, label='Observed')
        
        for m_name, m_cfg in MODELS.items():
            func = m_cfg['func']
            try:
                # 针对不同模型设置 p0 和 bounds
                if m_name == 'Inv_Gaussian':
                    p0 = [ti_max, ti_max-ti_min, z_at_min, 50]
                    bounds = (-np.inf, np.inf)
                elif m_name == 'Inv_Asym_Gauss':
                    p0 = [ti_max, ti_max-ti_min, z_at_min, 40, 60]
                    bounds = (-np.inf, np.inf)
                elif m_name == 'Inv_Banta':
                    p0 = [ti_max, ti_max-ti_min, z_at_min, 1.0, 1.0]
                    bounds = ([0, 0, 50, 0.1, 0.1], [1, 1, 500, 10, 10])
                elif m_name == 'Inv_Sech':
                    p0 = [ti_max, ti_max-ti_min, z_at_min, 50]
                    bounds = ([0, 0, 50, 1], [1, 1, 500, 600])

                # 拟合
                popt, _ = curve_fit(func, z_vals, ti_vals, p0=p0, bounds=bounds if 'bounds' in locals() else (-np.inf, np.inf), maxfev=5000)
                
                # 计算指标
                y_pred = func(z_vals, *popt)
                rmse = np.sqrt(mean_squared_error(ti_vals, y_pred))
                r2 = r2_score(ti_vals, y_pred)
                
                # 记录到字典
                station_res[f'{m_name}_RMSE'] = rmse
                station_res[f'{m_name}_R2'] = r2
                
                # 更新最佳模型
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_name = m_name
                
                # 绘图
                y_smooth = func(z_smooth, *popt)
                # 线宽：最佳模型加粗
                lw = 1.5
                ax.plot(y_smooth, z_smooth, color=m_cfg['c'], linestyle=m_cfg['ls'], linewidth=lw, alpha=0.8, label=m_name)
                
            except Exception as e:
                # print(f" {m_name}失败", end='')
                station_res[f'{m_name}_RMSE'] = np.nan
        
        print(f" 完成 | 最佳: {best_model_name}")
        station_res['Winner'] = best_model_name
        summary_results.append(station_res)
        
        # 子图修饰
        title_str = f"{site_name}\nBest: {best_model_name}"
        ax.set_title(title_str, fontsize=10, fontweight='bold', color='darkred')
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(fontsize=8) # 只在第一个图显示图例

    # 清理空白子图
    for i in range(idx + 1, len(axes_flat)):
        fig_all.delaxes(axes_flat[i])
        
    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, 'All_Stations_Models_Compare.png')
    plt.savefig(img_path, dpi=300)
    print(f"\n[图表] 10场站竞技对比图已保存: {img_path}")
    
    # --- 保存汇总 Excel ---
    if summary_results:
        df_sum = pd.DataFrame(summary_results)
        # 重新排序列，把 Winner 放在前面
        cols = ['Station', 'Winner'] + [c for c in df_sum.columns if c not in ['Station', 'Winner']]
        df_sum = df_sum[cols]
        
        excel_path = os.path.join(OUTPUT_DIR, 'Arena_Results_Summary.xlsx')
        df_sum.to_excel(excel_path, index=False)
        print(f"[数据] 误差对比表已保存: {excel_path}")
        
        # 简单的统计
        print("\n>>> 获胜统计 (Count of Best Model) <<<")
        print(df_sum['Winner'].value_counts())

if __name__ == "__main__":
    main()