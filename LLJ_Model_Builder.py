import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
import glob

# ================= 配置区域 =================
# 请确保这里的路径是正确的
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/model_build'

# LLJ 判定标准 (可根据需要微调)
LLJ_THRESHOLD = 2.0      # 鼻尖风速比上下边缘大 2m/s
MIN_JET_HEIGHT = 60     # 急流高度下限
MAX_JET_HEIGHT = 480     # 急流高度上限 
# ===========================================

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. 核心模型公式 (用于拟合归一化后的数据)
# ---------------------------------------------------------

def model_normalized_speed(z_norm, alpha, beta):
    """
    通用急流模型 (基于 Banta et al., 2006 / Slab model 变体)
    z_norm = z / Z_jet
    u_norm = U / U_jet
    理论上当 z_norm = 1 时，函数值应约为 1
    """
    # 公式： U_norm = (z_norm)^alpha * exp(beta * (1 - z_norm))
    return np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))

# ---------------------------------------------------------
# 2. 数据读取与处理 (智能编码版)
# ---------------------------------------------------------

def read_windographer_txt(file_path):
    """
    智能读取 Windographer 导出的 txt 文件 (增强编码兼容性)
    自动轮询 utf-8, gbk 等常见编码，解决中文乱码问题
    """
    filename = os.path.basename(file_path)
    # print(f"正在读取: {filename} ...")
    
    # 尝试的编码列表
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'gbk', 'utf-16', 'latin-1']
    
    header_row = -1
    encoding_used = None
    
    # 1. 寻找表头行
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                # 读取前 50 行找表头
                lines = [f.readline() for _ in range(50)]
                
            for i, line in enumerate(lines):
                # 关键判定：同时包含英文和中文关键词，确保没有乱码
                if "Date/Time" in line and "m水平风速" in line:
                    header_row = i
                    encoding_used = enc
                    break
            
            if header_row != -1:
                # print(f"  -> 识别到编码: {enc}, 表头在第 {header_row+1} 行")
                break 
                
        except UnicodeDecodeError:
            continue # 如果这个编码读都读不了，就换下一个
        except Exception:
            continue

    if header_row == -1:
        print(f"  [跳过] {filename}: 未找到包含 'Date/Time' 和 'm水平风速' 的表头。")
        return None

    # 2. 读取数据 (使用确认的编码)
    try:
        # 使用 python 引擎更稳定
        df = pd.read_csv(file_path, skiprows=header_row, sep='\t', encoding=encoding_used, engine='python')
    except:
        try:
            # 如果 \t 分隔失败，尝试空格分隔
            df = pd.read_csv(file_path, skiprows=header_row, sep='\s+', encoding=encoding_used, engine='python')
        except Exception as e:
            print(f"  [错误] Pandas 读取失败: {e}")
            return None
        
    # 3. 清洗列名 (去空格和引号)
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    
    return df

def extract_profiles(df):
    """
    从 DataFrame 中提取高度层数据，并计算 TI
    """
    # 识别高度
    cols = df.columns
    # 筛选风速列 (排除最大、最小等统计列)
    speed_cols = [c for c in cols if 'm水平风速' in c and '最大' not in c and '最小' not in c and '偏差' not in c]
    
    # 提取数字高度
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    if not heights: return None, None, None
    
    # 构建矩阵: Rows=Time, Cols=Height
    n_samples = len(df)
    n_heights = len(heights)
    
    ws_matrix = np.full((n_samples, n_heights), np.nan)
    ti_matrix = np.full((n_samples, n_heights), np.nan)
    
    for i, h in enumerate(heights):
        # 1. 找风速列
        ws_col_list = [c for c in cols if f'{h}m水平风速' in c and '最大' not in c and '最小' not in c and '偏差' not in c]
        if not ws_col_list: continue
        ws_col = ws_col_list[0]
        
        # 2. 找标准差/偏差列 (用于计算 TI)
        # 模糊匹配：包含高度h，且包含 '偏差' 或 'Std'，且不包含 '风向'
        std_col_candidates = [c for c in cols if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c]
        
        # 提取数据
        ws_vals = pd.to_numeric(df[ws_col], errors='coerce').values
        ws_matrix[:, i] = ws_vals
        
        if std_col_candidates:
            std_vals = pd.to_numeric(df[std_col_candidates[0]], errors='coerce').values
            # 计算 TI = Std / Mean
            with np.errstate(divide='ignore', invalid='ignore'):
                ti_vals = std_vals / ws_vals
                # 风速太小 TI 不准，过滤掉 < 3.0 m/s 的数据
                ti_vals[ws_vals < 3.0] = np.nan 
            ti_matrix[:, i] = ti_vals
            
    return np.array(heights), ws_matrix, ti_matrix

# ---------------------------------------------------------
# 3. 主逻辑：归一化判定与建模
# ---------------------------------------------------------

def main():
    print("="*60)
    print(" LLJ 通用模型构建器 (Universal LLJ Model Builder)")
    print("="*60)
    
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    if not files:
        print(f"[错误] 在 {DATA_DIR} 下未找到 .txt 文件！")
        return

    print(f"检测到 {len(files)} 个文件，开始处理...")

    # 容器：存储所有场站、所有时刻识别出的急流数据
    all_profiles_raw = []  # 存原始风速廓线 [u_array]
    all_profiles_norm = [] # 存归一化坐标 [z_norm, u_norm]
    all_ti_norm = []       # 存归一化TI [z_norm, ti_val]
    z_vec_ref = None       # 记录一个参考高度向量用于画图

    # --- 循环处理每个文件 ---
    for f in files:
        df = read_windographer_txt(f)
        if df is None: continue
        
        z_vec, ws_mat, ti_mat = extract_profiles(df)
        if z_vec is None: continue
        if z_vec_ref is None: z_vec_ref = z_vec # 记录第一次的高度层
        
        # 逐行识别 LLJ
        count_local = 0
        for i in range(len(ws_mat)):
            u = ws_mat[i, :]
            ti = ti_mat[i, :]
            
            if np.isnan(u).any(): continue
            
            # 找到最大值位置
            idx_max = np.argmax(u)
            z_jet = z_vec[idx_max]
            u_jet = u[idx_max]
            
            # 判定标准
            # 1. 高度在范围内
            if not (MIN_JET_HEIGHT <= z_jet <= MAX_JET_HEIGHT): continue
            
            # 2. 上下切变强度 (Nose shape)
            u_bottom = u[0]
            u_top = u[-1]
            if (u_jet - u_bottom < LLJ_THRESHOLD) or (u_jet - u_top < LLJ_THRESHOLD):
                continue
                
            # --- 这是一个有效的急流事件 ---
            count_local += 1
            
            # 1. 存储原始数据 (插值到统一高度很难，这里暂存原始值用于概略绘图)
            # 为了画平均图，我们需要统一高度。这里简单处理：只存高度层和 z_vec_ref 一致的数据
            if len(u) == len(z_vec_ref) and np.array_equal(z_vec, z_vec_ref):
                all_profiles_raw.append(u)
            
            # 2. 存储归一化数据 (核心)
            z_norm = z_vec / z_jet
            u_norm = u / u_jet
            
            all_profiles_norm.append([z_norm, u_norm])
            
            # 3. 存储 TI 数据
            if not np.isnan(ti).all():
                all_ti_norm.append([z_norm, ti])
        
        # print(f"  -> {os.path.basename(f)}: 提取到 {count_local} 个急流样本")

    # 检查是否有数据
    if not all_profiles_norm:
        print("[错误] 所有文件中均未识别到符合条件的急流事件。请检查阈值设置。")
        return

    n_total = len(all_profiles_norm)
    print(f"\n[汇总] 共提取到 {n_total} 个高质量急流样本。正在进行归一化建模...")

    # ---------------------------------------------------------
    # 4. 可视化判定：为什么要归一化？
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 随机抽样绘图 (避免几万条线画死)
    sample_indices = np.random.choice(len(all_profiles_norm), size=min(300, len(all_profiles_norm)), replace=False)
    
    # --- 图 1：物理坐标 (Direct Averaging 的问题) ---
    # 只画那些高度层一致的数据
    if all_profiles_raw:
        raw_array = np.array(all_profiles_raw)
        # 画背景线
        n_raw_sample = min(300, len(raw_array))
        idx_raw = np.random.choice(len(raw_array), size=n_raw_sample, replace=False)
        for i in idx_raw:
            axes[0].plot(raw_array[i], z_vec_ref, color='gray', alpha=0.05)
        
        # 画平均线
        mean_raw = np.nanmean(raw_array, axis=0)
        axes[0].plot(mean_raw, z_vec_ref, color='red', linewidth=3, label='Arithmetic Mean')
        
        axes[0].set_title(f'(A) Physical Coordinates\n(Different Jet Heights blur the nose)', fontsize=14)
        axes[0].set_xlabel('Wind Speed [m/s]')
        axes[0].set_ylabel('Height [m]')
        axes[0].legend()
        axes[0].grid(True)
    else:
        axes[0].text(0.5, 0.5, "Inconsistent heights across files\nCannot plot raw average", ha='center')

    # --- 图 2：归一化坐标 (Normalized) ---
    all_z_norm_points = []
    all_u_norm_points = []
    
    for i in range(len(all_profiles_norm)):
        zn, un = all_profiles_norm[i]
        all_z_norm_points.extend(zn)
        all_u_norm_points.extend(un)
        # 绘图只画样本
        if i in sample_indices:
            axes[1].plot(un, zn, color='blue', alpha=0.05)

    all_z_norm_points = np.array(all_z_norm_points)
    all_u_norm_points = np.array(all_u_norm_points)

    # ---------------------------------------------------------
    # 5. 拟合模型
    # ---------------------------------------------------------
    print(" -> 正在拟合通用风速廓线参数...")
    mask = np.isfinite(all_z_norm_points) & np.isfinite(all_u_norm_points)
    
    try:
        # 拟合通用公式
        popt, pcov = curve_fit(model_normalized_speed, 
                               all_z_norm_points[mask], 
                               all_u_norm_points[mask], 
                               p0=[2.0, 2.0], bounds=([0, 0], [10, 10]))
        
        alpha_fit, beta_fit = popt
        
        # 画出拟合曲线
        z_fit = np.linspace(0, 2.5, 100) # 归一化高度 z/Z_jet 从 0 到 2.5
        u_fit = model_normalized_speed(z_fit, *popt)
        
        axes[1].plot(u_fit, z_fit, color='red', linewidth=3, label=f'Model Fit\nα={alpha_fit:.2f}, β={beta_fit:.2f}')
        axes[1].set_title(f'(B) Normalized Coordinates\n(Structure Collapsed!)', fontsize=14)
        axes[1].set_xlabel('Normalized Speed (u/U_jet)')
        axes[1].set_ylabel('Normalized Height (z/Z_jet)')
        axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.3) 
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_ylim(0, 2.5) # 限制Y轴范围
        
    except Exception as e:
        print(f"[错误] 拟合失败: {e}")
        return

    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, 'Normalization_Proof.png')
    plt.savefig(out_img, dpi=300)
    print(f"[图表] 归一化验证图已保存: {out_img}")

    # ---------------------------------------------------------
    # 6. 输出最终结论
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" >>> 最终风模型参数 (Final Wind Model Parameters) <<<")
    print("="*60)
    print("基于所有场站数据的归一化分析，推荐使用以下通用模型：")
    print("\n[公式形式]")
    print("  U(z) = U_jet * (z/Z_jet)^α * exp( β * (1 - z/Z_jet) )")
    print("\n[拟合参数]")
    print(f"  α (Alpha) = {alpha_fit:.4f}")
    print(f"  β (Beta)  = {beta_fit:.4f}")
    
    print("-" * 60)
    print("【如何使用此模型？】")
    print("1. 确定你想要模拟的急流工况，例如：")
    print("   - 急流核心高度 Z_jet = 160 m")
    print("   - 核心最大风速 U_jet = 12.0 m/s")
    print("2. 将上述参数代入公式：")
    print(f"   U(z) = 12.0 * (z/160)^{alpha_fit:.2f} * exp({beta_fit:.2f} * (1 - z/160))")
    print("3. 计算任意高度 z 处的风速即可得到完整的物理廓线。")
    print("="*60)

if __name__ == "__main__":
    main()