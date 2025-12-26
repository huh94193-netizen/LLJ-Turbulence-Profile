import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm
import re
import os
import glob

# ================= 配置区域 =================
INPUT_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result'
IMG_DIR = os.path.join(OUTPUT_DIR, 'images')

# 是否强制分类数量？ (None表示自动判断，填 2 或 3 表示强制)
FORCE_N_CLUSTERS = 3 

# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# --- 1. 解决字体问题的核心函数 ---
def get_chinese_font():
    """尝试找到系统可用的中文字体，找不到则返回 None"""
    # 常见 Linux/Windows 中文字体路径
    font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    for font in font_names:
        try:
            # 检查 matplotlib 是否能找到该字体
            if fm.findfont(font) != fm.findfont('DejaVu Sans'): # 如果找到了非默认字体
                return font
        except: continue
    return None

# 设置绘图字体
chosen_font = get_chinese_font()
if chosen_font:
    plt.rcParams['font.sans-serif'] = [chosen_font]
    plt.rcParams['axes.unicode_minus'] = False
    print(f"[字体] 已启用中文字体: {chosen_font}")
else:
    print("[字体] 未找到中文字体，图表标签将使用 Site_ID 代替中文名以避免乱码。")

# --- 2. 数据解析函数 (保持不变) ---
def strict_tab_parse_v3(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'gbk', 'latin-1']
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

def vector_mean_direction(dir_degrees):
    rads = np.radians(dir_degrees)
    mean_u = np.nanmean(np.sin(rads))
    mean_v = np.nanmean(np.cos(rads))
    mean_deg = np.degrees(np.arctan2(mean_u, mean_v))
    if mean_deg < 0: mean_deg += 360
    return mean_deg

def gaussian_jet_model(z, u_base, u_jet, z_jet, sigma):
    return u_base + u_jet * np.exp(-((z - z_jet)**2) / (2 * sigma**2))

def power_law(z, alpha, u_ref):
    return u_ref * (z / z_ref_fixed)**alpha

# --- 3. 分析逻辑 ---
def analyze_single_site(file_path, site_id):
    filename = os.path.basename(file_path)
    # 尝试提取中文名，如果提取不到就用文件名
    site_name = filename.split('-')[0]
    
    # 如果没有中文字体，强制把 site_name 改成 ID，避免画图方框
    display_name = site_name if chosen_font else f"Site_{site_id}"
    
    print(f" -> 处理: {site_name} (ID: {display_name})")
    
    df_raw = strict_tab_parse_v3(file_path)
    if df_raw is None: return None, None, None

    df_raw.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df_raw.columns]
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c and '最小' not in c and '偏差' not in c]
    if not speed_cols: return None, None, None
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])

    data_dict = {}
    for h in heights:
        col = f'{h}m水平风速'
        if col in df_raw.columns:
            data_dict[f'ws_{h}'] = pd.to_numeric(df_raw[col], errors='coerce')
    
    if '500m数据可靠性' in df_raw.columns:
        data_dict['avail'] = pd.to_numeric(df_raw['500m数据可靠性'], errors='coerce')
        
    df_calc = pd.DataFrame(data_dict)
    if 'avail' in df_calc: df_calc = df_calc[df_calc['avail'] >= 80.0]
    
    ws_cols = [f'ws_{h}' for h in heights]
    df_clean = df_calc.dropna(subset=ws_cols)
    speed_mat = df_clean[ws_cols].values
    valid_idx = df_clean.index
    llj_indices = []
    
    for i in range(len(speed_mat)):
        s = speed_mat[i]
        try:
            mx_i = np.nanargmax(s)
            mx_v = s[mx_i]
            mx_h = heights[mx_i]
            if mx_h <= 100 or mx_h >= 480: continue
            if (mx_v - s[0] >= 2.0) and (mx_v - s[-1] >= 2.0):
                llj_indices.append(valid_idx[i])
        except: continue
        
    if len(llj_indices) < 5: return None, None, None

    df_ev = df_raw.loc[llj_indices]
    res_h, res_ws, res_ti, res_wd = [], [], [], []
    
    for h in heights:
        ws = pd.to_numeric(df_ev[f'{h}m水平风速'], errors='coerce')
        res_h.append(h)
        res_ws.append(ws.mean())
        
        std_col = next((c for c in df_ev.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)
        if std_col:
            std_val = pd.to_numeric(df_ev[std_col], errors='coerce')
            ti = std_val / ws
            ti[ws < 3.0] = np.nan
            res_ti.append(ti.mean())
        else: res_ti.append(np.nan)
        
        wd_col = next((c for c in df_ev.columns if str(h) in c and ('风向' in c) and '最大' not in c), None)
        if wd_col:
            wd = pd.to_numeric(df_ev[wd_col], errors='coerce')
            res_wd.append(vector_mean_direction(wd.values))
        else: res_wd.append(np.nan)

    z_vals = np.array(res_h)
    u_vals = np.array(res_ws)
    
    feat_jet_height = np.nan
    feat_max_speed = np.nan
    feat_alpha = np.nan
    
    # 拟合
    try:
        p0 = [5, 5, 260, 100]
        popt, _ = curve_fit(gaussian_jet_model, z_vals, u_vals, p0=p0, maxfev=5000)
        feat_jet_height = popt[2]
        feat_max_speed = gaussian_jet_model(popt[2], *popt)
    except: pass
    
    try:
        idx_max = np.argmax(u_vals)
        z_peak = z_vals[idx_max]
        mask = z_vals <= z_peak
        if np.sum(mask) > 3:
            global z_ref_fixed
            z_ref_fixed = z_vals[0]
            popt_p, _ = curve_fit(power_law, z_vals[mask], u_vals[mask])
            feat_alpha = popt_p[0]
    except: pass

    # 简单绘图
    plt.figure(figsize=(6, 8))
    plt.plot(res_ws, res_h, 'o-')
    plt.title(f'{display_name} Profile') # 使用 display_name (可能是ID)
    plt.savefig(os.path.join(IMG_DIR, f'{display_name}_Profile.png'))
    plt.close()

    df_out = pd.DataFrame({'Height': res_h, 'Speed': res_ws, 'TI': res_ti, 'Dir': res_wd})
    
    feature_dict = {
        'Site_ID': display_name,      # 用于绘图的简短ID
        'Real_Name': site_name,       # 真实的中文名
        'Jet_Height': feat_jet_height,
        'Max_Speed': feat_max_speed,
        'Alpha': feat_alpha
    }
    
    return site_name, df_out, feature_dict

# --- 4. 主程序 ---
def main():
    print("="*60)
    files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))
    excel_path = os.path.join(OUTPUT_DIR, 'All_Sites_Data_V11.xlsx')
    feature_list = []
    
    with pd.ExcelWriter(excel_path) as writer:
        for idx, f in enumerate(files):
            try:
                name, df_res, features = analyze_single_site(f, idx+1)
                if df_res is not None:
                    # Excel Sheet名不能有特殊字符，用 Site_ID
                    df_res.to_excel(writer, sheet_name=features['Site_ID'], index=False)
                    feature_list.append(features)
            except Exception as e:
                print(f"    [错误] {e}")

    # 聚类
    df_feat = pd.DataFrame(feature_list)
    df_model = df_feat.dropna(subset=['Jet_Height', 'Max_Speed', 'Alpha']).copy()
    
    # 导出特征表 (关键修复：utf-8-sig)
    csv_path = os.path.join(OUTPUT_DIR, 'Site_Features_Fixed.csv')
    df_model.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[成功] 特征表已保存 (乱码已修复): {csv_path}")

    if len(df_model) < 2: return

    X = df_model[['Jet_Height', 'Max_Speed', 'Alpha']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 聚类数判定
    if FORCE_N_CLUSTERS:
        k = FORCE_N_CLUSTERS
    else:
        k = 3 if len(df_model) >= 6 else 2
        
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_model['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 保存带聚类结果的表
    df_model.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 绘图
    plt.figure(figsize=(12, 9))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    
    for c in range(k):
        cluster_data = df_model[df_model['Cluster'] == c]
        plt.scatter(cluster_data['Jet_Height'], cluster_data['Max_Speed'], 
                    s=200, c=colors[c], label=f'Cluster {c+1}', edgecolors='k', alpha=0.8)
        
        # 标注 (使用 Site_ID 避免乱码)
        for idx, row in cluster_data.iterrows():
            # 标注格式: ID (Alpha)
            label_text = f"{row['Site_ID']}\n(α={row['Alpha']:.2f})"
            plt.text(row['Jet_Height']+1, row['Max_Speed'], label_text, fontsize=10)

    plt.title(f'LLJ Clustering (K={k})\nLabels show: Site_ID (Shear Alpha)', fontsize=16)
    plt.xlabel('Jet Core Height [m]', fontsize=14)
    plt.ylabel('Jet Max Speed [m/s]', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    out_img = os.path.join(OUTPUT_DIR, 'Cluster_Final_Fixed.png')
    plt.savefig(out_img, dpi=300)
    print(f"[成功] 聚类图已保存: {out_img}")
    print("  -> 图中 ID 对应的真实中文名请查看 Site_Features_Fixed.csv")

if __name__ == "__main__":
    main()