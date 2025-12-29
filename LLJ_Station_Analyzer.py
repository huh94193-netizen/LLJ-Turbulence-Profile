import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
import glob
import matplotlib.font_manager as fm

# --- 核心修复：解决 Linux 下 Matplotlib 中文乱码 ---
def configure_chinese_font():
    """
    尝试自动寻找并设置系统可用的中文字体
    """
    # Linux/Windows 常见中文字体列表 (优先级从高到低)
    font_candidates = [
        'WenQuanYi Micro Hei',    # Linux 常见
        'Zen Hei',                # Linux 常见
        'Droid Sans Fallback',    # Linux 常见
        'SimHei',                 # Windows 默认
        'Microsoft YaHei',        # Windows 默认
        'SimSun',                 # Windows 宋体
        'Noto Sans CJK SC',       # Google 开源
    ]
    
    found_font = None
    for font_name in font_candidates:
        try:
            # 检查系统是否安装了该字体
            if fm.findfont(font_name) != fm.findfont('DejaVu Sans'): # 确保不是 fallback 到默认
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
                print(f"[字体] 成功启用中文字体: {font_name}")
                found_font = font_name
                break
        except:
            continue
            
    if not found_font:
        print("[警告] 未找到中文字体，图例可能显示乱码。建议改用英文 ID。")
        # 找不到字体时的保底策略：打印所有可用字体供调试 (可选)
        # print("可用字体:", [f.name for f in fm.fontManager.ttflist])

# 在主程序开始前运行一次配置
configure_chinese_font()
# ================= 配置区域 =================
DATA_DIR = r'/home/huxun/02_LLJ/exported_data'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/station_compare'

# 判定标准
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. 定义模型 ---
def model_normalized_speed(z_norm, alpha, beta):
    # 为防止计算溢出，做一点保护
    return np.power(z_norm, alpha) * np.exp(beta * (1.0 - z_norm))

# --- 2. 智能读取 (复用之前的稳健版) ---
def read_windographer_txt(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'gbk', 'utf-16', 'latin-1']
    header_row = -1
    encoding_used = None
    
    # 找表头
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                lines = [f.readline() for _ in range(50)]
            for i, line in enumerate(lines):
                if "Date/Time" in line and "m水平风速" in line:
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

def extract_profiles_and_norm(df):
    """提取急流并归一化"""
    cols = df.columns
    speed_cols = [c for c in cols if 'm水平风速' in c and '最大' not in c and '最小' not in c and '偏差' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    if not heights: return None, None
    
    # 构建矩阵
    n = len(df)
    ws_mat = np.full((n, len(heights)), np.nan)
    for i, h in enumerate(heights):
        col = [c for c in cols if f'{h}m水平风速' in c and '最大' not in c][0]
        ws_mat[:, i] = pd.to_numeric(df[col], errors='coerce').values
        
    # 识别急流
    norm_z = []
    norm_u = []
    
    valid_count = 0
    for i in range(n):
        u = ws_mat[i, :]
        if np.isnan(u).any(): continue
        
        mx_i = np.argmax(u)
        z_jet = heights[mx_i]
        u_jet = u[mx_i]
        
        if not (MIN_JET_HEIGHT <= z_jet <= MAX_JET_HEIGHT): continue
        if (u_jet - u[0] < LLJ_THRESHOLD) or (u_jet - u[-1] < LLJ_THRESHOLD): continue
        
        # 归一化
        norm_z.extend(np.array(heights) / z_jet)
        norm_u.extend(u / u_jet)
        valid_count += 1
        
    return np.array(norm_z), np.array(norm_u), valid_count

# --- 3. 主逻辑 ---
def main():
    print("="*60)
    print(" 分场站急流模型参数大比武")
    print("="*60)
    
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    results = [] # 存结果表
    
    # 设置绘图 - 我们把所有拟合线画在一张图上看区别
    plt.figure(figsize=(10, 8))
    z_fit_line = np.linspace(0, 2.5, 100)
    
    # 颜色库
    colors = plt.cm.jet(np.linspace(0, 1, len(files)))

    for idx, f in enumerate(files):
        # 提取场站名 (文件名第一段)
        station_name = os.path.basename(f).split('-')[0]
        print(f"[{idx+1}/{len(files)}] 处理: {station_name} ...", end='')
        
        df = read_windographer_txt(f)
        if df is None: 
            print(" 读取失败")
            continue
            
        nz, nu, count = extract_profiles_and_norm(df)
        
        if count < 10:
            print(f" 样本不足 ({count})")
            continue
            
        # 拟合
        try:
            # 初始猜测给 1.0, 1.0
            popt, _ = curve_fit(model_normalized_speed, nz, nu, p0=[1.0, 1.0], bounds=([0, 0], [10, 10]))
            alpha, beta = popt
            
            # 计算拟合优度 (简单 RMSE)
            u_pred = model_normalized_speed(nz, *popt)
            rmse = np.sqrt(np.mean((nu - u_pred)**2))
            
            print(f" -> Alpha={alpha:.3f}, Beta={beta:.3f}, Samples={count}")
            
            results.append({
                'Station': station_name,
                'Samples': count,
                'Alpha': alpha,
                'Beta': beta,
                'Shape_Factor': (alpha + beta)/2, # 用均值代表"尖锐度"
                'RMSE': rmse
            })
            
            # 绘图
            u_fit_line = model_normalized_speed(z_fit_line, alpha, beta)
            plt.plot(u_fit_line, z_fit_line, color=colors[idx], linewidth=2, label=f'{station_name}')
            
        except Exception as e:
            print(f" 拟合出错: {e}")

    # --- 总结与保存 ---
    if not results: return
    
    # 1. 保存 Excel 表格
    df_res = pd.DataFrame(results)
    # 按尖锐度排序
    df_res = df_res.sort_values('Shape_Factor', ascending=False)
    
    excel_path = os.path.join(OUTPUT_DIR, 'Station_Parameters_Compare.xlsx')
    df_res.to_excel(excel_path, index=False)
    print(f"\n[完成] 参数对比表已保存: {excel_path}")
    print(df_res[['Station', 'Alpha', 'Beta', 'Samples']]) # 打印预览

    # 2. 保存对比图
    plt.title('Normalized LLJ Shape Comparison by Station')
    plt.xlabel('Normalized Speed (U/U_jet)')
    plt.ylabel('Normalized Height (Z/Z_jet)')
    plt.axhline(1.0, color='k', linestyle=':', alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 图例放外边
    plt.tight_layout()
    
    img_path = os.path.join(OUTPUT_DIR, 'Station_Shape_Compare.png')
    plt.savefig(img_path, dpi=300)
    print(f"[完成] 对比图已保存: {img_path}")
    
    # 3. 聚类建议
    print("\n>>> 分析建议 <<<")
    print("请查看生成的图片和表格：")
    print("1. 如果所有曲线重合度很高，说明虽然地点不同，但急流的【垂直结构】是相似的，可以用统一模型。")
    print("2. 如果曲线明显分为几簇（比如有的很胖，有的很瘦），建议按地域分组（如：沿海组、平原组）分别给出公式。")

if __name__ == "__main__":
    main()