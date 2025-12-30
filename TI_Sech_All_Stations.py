import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.font_manager as fm
import re
import os
import glob
import warnings

# ================= 配置区域 =================
INPUT_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/ti_sech_all_stations'

# 质量控制
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 60
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
                # print(f"[系统] 字体已启用: {font_name}")
                return font_name
        except: continue
    return None

sys_font = configure_chinese_font()

# --- 1. 定义核心模型: Inverted Sech ---
def model_inv_sech(z, ti_base, ti_dip, z_jet, width):
    """
    倒置双曲正割 (Inverted Sech)
    TI(z) = Base - Dip * sech( (z - Z_jet) / Width )
    """
    arg = (z - z_jet) / width
    # 使用 1/cosh 避免 numpy 没有 sech 的问题
    # 增加保护防止溢出
    val = np.zeros_like(arg)
    mask = np.abs(arg) < 700 # cosh(700) 接近 float64 上限
    val[mask] = 1.0 / np.cosh(arg[mask])
    # 超过范围的 sech 趋近于 0，保持为 0 即可
    
    return ti_base - ti_dip * val

# --- 2. 数据读取与处理 ---
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

    # 清洗列名
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    n = len(df_raw)
    ws_mat = np.full((n, len(heights)), np.nan)
    ti_mat = np.full((n, len(heights)), np.nan)
    
    for i, h in enumerate(heights):
        ws_c = [c for c in df_raw.columns if f'{h}m水平风速' in c and '最大' not in c][0]
        # 找 TI (偏差/Std)
        std_c_list = [c for c in df_raw.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c]
        if std_c_list:
            std_c = std_c_list[0]
            w = pd.to_numeric(df_raw[ws_c], errors='coerce').values
            s = pd.to_numeric(df_raw[std_c], errors='coerce').values
            ws_mat[:, i] = w
            with np.errstate(divide='ignore', invalid='ignore'):
                ti = s / w
                ti[w < 3.0] = np.nan
            ti_mat[:, i] = ti
            
    # 筛选 LLJ
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

# --- 3. 批量处理主逻辑 ---
def main():
    print("="*60)
    print(" 批量处理：10场站 TI 倒置 Sech 模型拟合")
    print("="*60)
    
    files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))
    results = []
    
    # 准备 10合1 的大图 (假设最多12个文件，用 3x4 布局)
    n_files = len(files)
    rows = int(np.ceil(n_files / 4))
    fig_all, axes_all = plt.subplots(rows, 4, figsize=(20, 5 * rows))
    axes_flat = axes_all.flatten()
    
    for idx, f in enumerate(files):
        # 提取场站ID/名字
        filename = os.path.basename(f)
        try:
            # 尝试提取 "双鸭山集贤-1443#" 这种格式的前半部分
            site_name = filename.split('-')[0] + "-" + filename.split('-')[1]
        except:
            site_name = filename[:15]

        print(f"[{idx+1}/{n_files}] 处理: {site_name} ...", end='')
        
        z_vals, ti_vals = get_mean_ti_profile(f)
        ax = axes_flat[idx]
        
        if z_vals is None:
            print(" 跳过 (样本不足)")
            ax.text(0.5, 0.5, "Insufficient Data", ha='center')
            continue

        # --- 拟合 Sech ---
        try:
            ti_min = np.min(ti_vals)
            ti_max = np.max(ti_vals)
            z_at_min = z_vals[np.argmin(ti_vals)]
            
            # p0: [Base, Dip, Z_jet, Width]
            p0 = [ti_max, ti_max - ti_min, z_at_min, 50]
            # bounds: Width > 1
            bounds = ([0, 0, 50, 1], [1, 1, 500, 600])
            
            popt, _ = curve_fit(model_inv_sech, z_vals, ti_vals, p0=p0, bounds=bounds, maxfev=5000)
            
            # 计算 RMSE
            ti_pred = model_inv_sech(z_vals, *popt)
            rmse = np.sqrt(mean_squared_error(ti_vals, ti_pred))
            r2 = 1 - np.sum((ti_vals - ti_pred)**2) / np.sum((ti_vals - np.mean(ti_vals))**2)
            
            print(f" 成功 | RMSE={rmse:.5f}")
            
            # 记录结果
            results.append({
                'Station': site_name,
                'RMSE': rmse,
                'R2': r2,
                'TI_Base': popt[0],
                'TI_Dip': popt[1],
                'Z_Jet_Core': popt[2],
                'Width': popt[3]
            })
            
            # --- 绘图 (单站) ---
            z_smooth = np.linspace(z_vals[0], z_vals[-1], 200)
            ti_smooth = model_inv_sech(z_smooth, *popt)
            
            # 在大图上画
            ax.plot(ti_vals, z_vals, 'ko', markersize=4, alpha=0.6, label='Obs')
            ax.plot(ti_smooth, z_smooth, 'r-', linewidth=2, label='Sech Fit')
            
            # 如果没有中文字体，标题用 ID 防止乱码
            title_text = site_name if sys_font else "Station " + filename.split('-')[1]
            ax.set_title(f"{title_text}\nZ={popt[2]:.0f}m, W={popt[3]:.0f}m", fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0: ax.legend()
            
            # --- 保存单站精细图 (可选) ---
            # plt.figure()... (如果需要单张大图可以在这里加)
            
        except Exception as e:
            print(f" 拟合失败: {e}")
            ax.text(0.5, 0.5, "Fit Failed", ha='center')

    # 清理多余的子图
    for i in range(idx + 1, len(axes_flat)):
        fig_all.delaxes(axes_flat[i])
        
    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, 'All_Stations_Sech_Fit.png')
    plt.savefig(out_img, dpi=300)
    print(f"\n[图表] 汇总拼图已保存: {out_img}")
    
    # --- 保存 Excel ---
    if results:
        df_res = pd.DataFrame(results)
        # 调整列顺序
        cols = ['Station', 'RMSE', 'R2', 'TI_Base', 'TI_Dip', 'Z_Jet_Core', 'Width']
        df_res = df_res[cols]
        
        out_excel = os.path.join(OUTPUT_DIR, 'TI_Sech_Parameters_All.xlsx')
        df_res.to_excel(out_excel, index=False)
        print(f"[数据] 参数汇总表已保存: {out_excel}")
        
        # 打印预览
        print("\n>>> 参数预览 (前5行) <<<")
        print(df_res.head().to_string(index=False))

if __name__ == "__main__":
    main()