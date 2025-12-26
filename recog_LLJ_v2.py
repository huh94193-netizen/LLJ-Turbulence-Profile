import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os

# ================= 配置区域 =================
file_path = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result'

# 1. 质量控制
MIN_AVAILABILITY = 80.0 

# 2. 急流判定
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

def strict_tab_parse_v3(file_path):
    print(f" -> 启动解析...")
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'latin-1']
    raw_lines = []
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                raw_lines = f.readlines()
            break
        except:
            continue
    
    if not raw_lines: return None

    header_idx = -1
    header_parts = []
    for i, line in enumerate(raw_lines[:100]):
        # 只要包含 Date/Time 或者是 m水平风速 (V5逻辑)
        if "Date/Time" in line or "m水平风速" in line:
            header_idx = i
            header_parts = line.strip().split('\t')
            break
            
    if header_idx == -1: return None
    header_parts = [h.strip().replace('"', '') for h in header_parts]

    data_list = []
    for i in range(header_idx + 1, len(raw_lines)):
        line = raw_lines[i].strip()
        if not line: continue
        parts = line.split('\t')
        parts = [p.strip().replace('"', '') for p in parts]
        if len(parts) > len(header_parts): parts = parts[:len(header_parts)]
        elif len(parts) < len(header_parts): parts += [''] * (len(header_parts) - len(parts))
        if not parts[0]: continue
        data_list.append(parts)

    return pd.DataFrame(data_list, columns=header_parts)

def vector_mean_direction(dir_degrees):
    """计算风向的矢量平均"""
    rads = np.radians(dir_degrees)
    u = np.sin(rads)
    v = np.cos(rads)
    mean_u = np.nanmean(u)
    mean_v = np.nanmean(v)
    mean_deg = np.degrees(np.arctan2(mean_u, mean_v))
    if mean_deg < 0: mean_deg += 360
    return mean_deg

def gaussian_jet_model(z, u_base, u_jet, z_jet, sigma):
    return u_base + u_jet * np.exp(-((z - z_jet)**2) / (2 * sigma**2))

def power_law(z, alpha, u_ref):
    return u_ref * (z / z_ref_fixed)**alpha

def process_lidar_data(file_path):
    print("="*60)
    print(f"正在处理文件: {os.path.basename(file_path)}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 读取原始数据
    df_raw = strict_tab_parse_v3(file_path)
    if df_raw is None: return

    # 清洗列名
    df_raw.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df_raw.columns]
    
    # 2. 锁定列名 (核心修复：使用 V5 的严格匹配逻辑)
    # 只找包含 'm水平风速' 的列，排除其他杂项
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c and '最小' not in c and '偏差' not in c]
    
    if not speed_cols:
        print("[错误] 未找到 'm水平风速' 列。")
        return

    # 提取高度
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    print(f" -> 识别到的高度层: {heights}")

    # 3. 构建计算用的干净 DataFrame (解决碎片化警告)
    # 我们不修改 df_raw，而是把需要的列提取出来转成数字
    print(" -> 正在提取数值并清洗 (解决碎片化警告)...")
    
    data_dict = {}
    
    # 3.1 处理风速
    for h in heights:
        col_name = f'{h}m水平风速'
        if col_name in df_raw.columns:
            data_dict[f'ws_{h}'] = pd.to_numeric(df_raw[col_name], errors='coerce')
    
    # 3.2 处理数据可靠性 (如果有)
    avail_col = '500m数据可靠性'
    if avail_col in df_raw.columns:
        data_dict['availability'] = pd.to_numeric(df_raw[avail_col], errors='coerce')

    # 生成中间 DF
    df_calc = pd.DataFrame(data_dict)
    
    # 过滤可靠性
    if 'availability' in df_calc.columns:
        df_calc = df_calc[df_calc['availability'] >= MIN_AVAILABILITY]

    # 4. 筛选急流 (逻辑回归 V5)
    print(" -> 正在筛选急流样本...")
    
    # 构造风速矩阵列名列表
    ws_cols_sorted = [f'ws_{h}' for h in heights]
    
    # 去除空值
    df_clean = df_calc.dropna(subset=ws_cols_sorted)
    
    # 提取矩阵加速计算
    speed_matrix = df_clean[ws_cols_sorted].values
    valid_indices = df_clean.index # 保留原始索引以便回溯找风向/TI
    
    llj_indices = []
    
    # 遍历矩阵
    for i in range(len(speed_matrix)):
        speeds = speed_matrix[i]
        try:
            max_idx = np.nanargmax(speeds)
        except: continue
        
        max_v = speeds[max_idx]
        max_h = heights[max_idx]
        
        if max_h <= MIN_JET_HEIGHT or max_h >= MAX_JET_HEIGHT: continue
        
        v_bottom = speeds[0]
        v_top = speeds[-1]
        
        if (max_v - v_bottom >= LLJ_THRESHOLD) and (max_v - v_top >= LLJ_THRESHOLD):
            llj_indices.append(valid_indices[i]) # 记录原始索引
            
    print(f" -> 捕获急流样本数: {len(llj_indices)}")
    if len(llj_indices) < 5: return
    
    # 5. 回溯计算统计量 (TI 和 风向)
    # 使用 llj_indices 从 df_raw 中提取对应行
    df_events_raw = df_raw.loc[llj_indices]
    
    mean_speeds = []
    mean_tis = []
    mean_dirs = []
    
    print(" -> 正在计算统计廓线...")
    
    for h in heights:
        # --- 风速 ---
        ws_col = f'{h}m水平风速'
        # 既然能被筛选出来，风速列肯定在，且转数字没问题
        ws_series = pd.to_numeric(df_events_raw[ws_col], errors='coerce')
        mean_speeds.append(ws_series.mean())
        
        # --- TI (标准差/风速) ---
        # 模糊匹配找偏差列
        std_col = None
        for c in df_events_raw.columns:
            if str(h) in c and ('偏差' in c or '标准差' in c) and '风速' in c: # 确保是风速的偏差
                std_col = c
                break
        
        if std_col:
            std_series = pd.to_numeric(df_events_raw[std_col], errors='coerce')
            # 计算单点 TI
            ti_series = std_series / ws_series
            # 过滤低风速 TI
            ti_series[ws_series < 3.0] = np.nan
            mean_tis.append(ti_series.mean())
        else:
            mean_tis.append(np.nan)
            
        # --- 风向 (模糊匹配 + 矢量平均) ---
        wd_col = None
        # 查找包含 "风向" 的列，且必须包含高度
        for c in df_events_raw.columns:
            # 匹配 "30m...风向"
            if str(h) in c and ('风向' in c or 'Direction' in c) and '最大' not in c:
                wd_col = c
                break
        
        if wd_col:
            wd_series = pd.to_numeric(df_events_raw[wd_col], errors='coerce')
            mean_dirs.append(vector_mean_direction(wd_series.values))
        else:
            mean_dirs.append(np.nan)

    # 转 Array
    mean_speeds = np.array(mean_speeds)
    mean_tis = np.array(mean_tis)
    mean_dirs = np.array(mean_dirs)

    # 6. 拟合
    print(" -> 正在进行曲线拟合...")
    z_vals = np.array(heights)
    
    # 高斯
    u_fit_gauss = None
    gauss_text = ""
    try:
        p0 = [5, 5, 260, 100]
        popt, _ = curve_fit(gaussian_jet_model, z_vals, mean_speeds, p0=p0, maxfev=10000)
        u_fit_gauss = gaussian_jet_model(z_vals, *popt)
        gauss_text = f"Gaussian Fit:\nH={popt[2]:.0f}m, V={gaussian_jet_model(popt[2], *popt):.1f}m/s"
    except: pass

    # 幂律
    u_fit_power = None
    z_power = None
    alpha_text = ""
    try:
        idx_max = np.argmax(mean_speeds)
        z_peak = z_vals[idx_max]
        mask = z_vals <= z_peak
        if np.sum(mask) > 3:
            global z_ref_fixed
            z_ref_fixed = z_vals[0]
            popt_p, _ = curve_fit(power_law, z_vals[mask], mean_speeds[mask])
            alpha = popt_p[0]
            z_power = np.linspace(z_vals[0], z_peak, 50)
            u_fit_power = power_law(z_power, alpha, popt_p[1])
            alpha_text = f"Power Law (z<{z_peak}m):\nAlpha = {alpha:.3f}"
    except: pass

    # 7. 绘图
    fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
    
    # Speed
    ax = axes[0]
    ax.plot(mean_speeds, heights, 'o-', lw=2, label='Observed', color='#1f77b4')
    if u_fit_gauss is not None:
        ax.plot(u_fit_gauss, heights, '--', color='red', lw=2, label='Gaussian')
        ax.text(0.05, 0.95, gauss_text, transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8))
    if u_fit_power is not None:
        ax.plot(u_fit_power, z_power, ':', color='green', lw=3, label='Power Law')
        ax.text(0.05, 0.80, alpha_text, transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8), color='green')
    ax.set_title('Wind Speed Profile')
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Height [m]')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # Direction
    ax = axes[1]
    if not np.isnan(mean_dirs).all():
        ax.plot(mean_dirs, heights, 'o-', color='purple', lw=2)
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_title('Wind Direction (Vector)')
        ax.set_xlabel('Direction [°]')
        ax.grid(True, linestyle='--', alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No Direction Data", ha='center')

    # TI
    ax = axes[2]
    if not np.isnan(mean_tis).all():
        ax.plot(mean_tis, heights, 's--', color='#d62728', lw=2)
        ax.set_title('TI Profile')
        ax.set_xlabel('TI [-]')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle(f'LLJ Analysis (Samples: {len(llj_indices)})', fontsize=16)
    plt.tight_layout()
    
    out_png = os.path.join(OUTPUT_DIR, 'LLJ_Result_V8.png')
    plt.savefig(out_png, dpi=300)
    
    # Save CSV
    out_csv = os.path.join(OUTPUT_DIR, 'LLJ_Data_V8.csv')
    df_res = pd.DataFrame({'Height': heights, 'Speed': mean_speeds, 'TI': mean_tis, 'Direction': mean_dirs})
    df_res.to_csv(out_csv, index=False)

    print("="*60)
    print(f"[成功] 图片已保存: {out_png}")
    print(f"[成功] 数据已保存: {out_csv}")

if __name__ == "__main__":
    process_lidar_data(file_path)