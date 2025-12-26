import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import os
import glob

# ================= 配置区域 =================
INPUT_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result'
IMG_DIR = os.path.join(OUTPUT_DIR, 'images')

# 1. 质量控制
MIN_AVAILABILITY = 80.0 
# 2. 急流判定
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ----------------- 核心工具函数 (复用 V9) -----------------
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

# ----------------- 单个场站分析函数 -----------------
def analyze_single_site(file_path):
    site_name = os.path.basename(file_path).split('-')[0] # 提取 "双鸭山集贤"
    print(f" -> 处理场站: {site_name} ...")
    
    df_raw = strict_tab_parse_v3(file_path)
    if df_raw is None: return None, None, None

    # 清洗列名
    df_raw.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df_raw.columns]
    
    # 锁定列
    speed_cols = [c for c in df_raw.columns if 'm水平风速' in c and '最大' not in c and '最小' not in c and '偏差' not in c]
    if not speed_cols: return None, None, None
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])

    # 提取数据
    data_dict = {}
    for h in heights:
        col = f'{h}m水平风速'
        if col in df_raw.columns:
            data_dict[f'ws_{h}'] = pd.to_numeric(df_raw[col], errors='coerce')
    
    if '500m数据可靠性' in df_raw.columns:
        data_dict['avail'] = pd.to_numeric(df_raw['500m数据可靠性'], errors='coerce')
        
    df_calc = pd.DataFrame(data_dict)
    if 'avail' in df_calc: df_calc = df_calc[df_calc['avail'] >= MIN_AVAILABILITY]
    
    # 筛选急流
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
            if mx_h <= MIN_JET_HEIGHT or mx_h >= MAX_JET_HEIGHT: continue
            if (mx_v - s[0] >= LLJ_THRESHOLD) and (mx_v - s[-1] >= LLJ_THRESHOLD):
                llj_indices.append(valid_idx[i])
        except: continue
        
    if len(llj_indices) < 5: 
        print(f"    [跳过] 样本不足 ({len(llj_indices)})")
        return None, None, None

    # 统计计算
    df_ev = df_raw.loc[llj_indices]
    res_h, res_ws, res_ws_std, res_ti, res_wd = [], [], [], [], []
    
    for h in heights:
        # Speed
        ws = pd.to_numeric(df_ev[f'{h}m水平风速'], errors='coerce')
        res_h.append(h)
        res_ws.append(ws.mean())
        res_ws_std.append(ws.std())
        
        # TI
        std_col = next((c for c in df_ev.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)
        if std_col:
            std_val = pd.to_numeric(df_ev[std_col], errors='coerce')
            ti = std_val / ws
            ti[ws < 3.0] = np.nan
            res_ti.append(ti.mean())
        else: res_ti.append(np.nan)
        
        # Dir
        wd_col = next((c for c in df_ev.columns if str(h) in c and ('风向' in c) and '最大' not in c), None)
        if wd_col:
            wd = pd.to_numeric(df_ev[wd_col], errors='coerce')
            res_wd.append(vector_mean_direction(wd.values))
        else: res_wd.append(np.nan)

    # 拟合与特征提取
    z_vals = np.array(res_h)
    u_vals = np.array(res_ws)
    
    # 特征变量初始化
    feat_jet_height = np.nan
    feat_max_speed = np.nan
    feat_alpha = np.nan
    
    # 1. 高斯拟合
    u_fit_gauss = None
    try:
        p0 = [5, 5, 260, 100]
        popt, _ = curve_fit(gaussian_jet_model, z_vals, u_vals, p0=p0, maxfev=5000)
        u_fit_gauss = gaussian_jet_model(z_vals, *popt)
        feat_jet_height = popt[2] # 提取急流高度
        feat_max_speed = gaussian_jet_model(popt[2], *popt) # 提取最大风速
    except: pass
    
    # 2. 幂律拟合
    u_fit_power = None
    try:
        idx_max = np.argmax(u_vals)
        z_peak = z_vals[idx_max]
        mask = z_vals <= z_peak
        if np.sum(mask) > 3:
            global z_ref_fixed
            z_ref_fixed = z_vals[0]
            popt_p, _ = curve_fit(power_law, z_vals[mask], u_vals[mask])
            feat_alpha = popt_p[0] # 提取切变指数
            # 绘图用
            z_p_plot = np.linspace(z_vals[0], z_peak, 50)
            u_fit_power = power_law(z_p_plot, feat_alpha, popt_p[1])
    except: pass

    # 绘图 (简化版，只保存)
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    
    # Speed
    ax = axes[0]
    ax.fill_betweenx(res_h, np.array(res_ws)-np.array(res_ws_std), np.array(res_ws)+np.array(res_ws_std), alpha=0.2)
    ax.plot(res_ws, res_h, 'o-', label='Observed')
    if u_fit_gauss is not None: ax.plot(u_fit_gauss, res_h, 'r--', label='Gaussian')
    if u_fit_power is not None: ax.plot(u_fit_power, z_p_plot, 'g:', lw=3, label='PowerLaw')
    ax.set_title(f'{site_name} - Speed')
    ax.legend()
    
    # Dir
    ax = axes[1]
    if not np.isnan(res_wd).all():
        ax.plot(res_wd, res_h, 'o-', color='purple')
        ax.set_xlim(0, 360)
    ax.set_title('Direction')
    
    # TI
    ax = axes[2]
    if not np.isnan(res_ti).all():
        ax.plot(res_ti, res_h, 's-', color='red')
    ax.set_title('TI')
    
    plt.suptitle(f'{site_name} LLJ Analysis (N={len(llj_indices)})')
    plt.savefig(os.path.join(IMG_DIR, f'{site_name}_LLJ.png'), dpi=150)
    plt.close()

    # 返回结果
    # 1. Excel用的DataFrame
    df_out = pd.DataFrame({
        'Height': res_h, 'Speed_Mean': res_ws, 'Speed_Std': res_ws_std,
        'TI_Mean': res_ti, 'Dir_Mean': res_wd
    })
    
    # 2. 聚类用的特征字典
    feature_dict = {
        'Site': site_name,
        'Samples': len(llj_indices),
        'Jet_Height': feat_jet_height,
        'Max_Speed': feat_max_speed,
        'Alpha': feat_alpha
    }
    
    return site_name, df_out, feature_dict

# ----------------- 主程序：批量处理与聚类 -----------------
def main():
    print("="*60)
    print(" 启动批量 LLJ 分析与聚类系统 V10.0")
    print("="*60)
    
    files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))
    print(f"检测到 {len(files)} 个文件。")
    
    excel_path = os.path.join(OUTPUT_DIR, 'All_Sites_LLJ_Data.xlsx')
    feature_list = []
    
    # 1. 批量循环
    with pd.ExcelWriter(excel_path) as writer:
        for f in files:
            try:
                name, df_res, features = analyze_single_site(f)
                if df_res is not None:
                    # 写入 Excel Sheet
                    sheet_name = name[:30] # Excel sheet名限制31字符
                    df_res.to_excel(writer, sheet_name=sheet_name, index=False)
                    # 收集特征
                    feature_list.append(features)
            except Exception as e:
                print(f"    [错误] 处理 {os.path.basename(f)} 失败: {e}")
                
    print(f"\n[成功] Excel 汇总已保存: {excel_path}")
    print(f"[成功] 单场站图片已保存至: {IMG_DIR}")

    # 2. 聚类分析
    if len(feature_list) < 2:
        print("有效场站少于2个，无法进行聚类分析。")
        return

    print("\n" + "="*60)
    print(" 正在进行多场站特征聚类分析...")
    print("="*60)
    
    df_feat = pd.DataFrame(feature_list)
    # 清洗无效特征 (比如拟合失败的 NaN)
    df_model = df_feat[['Site', 'Jet_Height', 'Max_Speed', 'Alpha']].dropna()
    
    if len(df_model) < 2:
        print("有效特征数据不足。")
        return

    # 标准化数据 (因为高度是几百，风速是几，量纲不同)
    X = df_model[['Jet_Height', 'Max_Speed', 'Alpha']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means 聚类 (设 k=3，如果只有少于3个场站则 k=2)
    k = 3 if len(df_model) >= 6 else 2
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_model['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 打印分组结果
    print(f"建议将场站分为 {k} 类建模：")
    for c in range(k):
        sites = df_model[df_model['Cluster'] == c]['Site'].tolist()
        print(f"  [类别 {c+1}]: {sites}")
        # 计算该类的平均特征
        mean_h = df_model[df_model['Cluster'] == c]['Jet_Height'].mean()
        mean_v = df_model[df_model['Cluster'] == c]['Max_Speed'].mean()
        print(f"     -> 特征: 急流高度约 {mean_h:.0f}m, 核心风速约 {mean_v:.1f}m/s")

    # 3. 绘制聚类散点图
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    
    for c in range(k):
        cluster_data = df_model[df_model['Cluster'] == c]
        plt.scatter(cluster_data['Jet_Height'], cluster_data['Max_Speed'], 
                    s=150, c=colors[c], label=f'Cluster {c+1}', edgecolors='k', alpha=0.8)
        
        # 标注名字
        for idx, row in cluster_data.iterrows():
            plt.text(row['Jet_Height']+5, row['Max_Speed'], row['Site'], fontsize=9)

    plt.title(f'Multi-Site LLJ Clustering Analysis (K={k})', fontsize=16)
    plt.xlabel('Jet Core Height [m]', fontsize=12)
    plt.ylabel('Jet Max Speed [m/s]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    cluster_img = os.path.join(OUTPUT_DIR, 'Cluster_Analysis_Result.png')
    plt.savefig(cluster_img, dpi=300)
    print(f"\n[成功] 聚类分析图已保存: {cluster_img}")
    
    # 保存特征表
    df_model.to_csv(os.path.join(OUTPUT_DIR, 'Site_Features_Summary.csv'), index=False)

if __name__ == "__main__":
    main()