import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import re
import os

# ================= 配置区域 =================
FILE_PATH = r'/home/huxun/02_LLJ/exported_data/双鸭山集贤-1443#-20240506-20251222-filter-Exported.txt'
OUTPUT_DIR = r'/home/huxun/02_LLJ/result/ti_smart_anchor_v18'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 质量控制
LLJ_THRESHOLD = 2.0
MIN_JET_HEIGHT = 100
MAX_JET_HEIGHT = 480
# ===========================================

# --- 1. 数据准备 ---
def load_data_matrix(file_path):
    print(f"正在读取数据: {os.path.basename(file_path)}")
    # (简化的读取逻辑，同前)
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'latin-1']
    raw_lines = []
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f: raw_lines = f.readlines()
            break
        except: continue
    if not raw_lines: return None, None
    
    header_idx = -1
    for i, line in enumerate(raw_lines[:100]):
        if "m水平风速" in line: header_idx = i; break
    if header_idx == -1: return None, None
    
    header = raw_lines[header_idx].strip().split('\t')
    header = [h.strip().replace('"', '') for h in header]
    data = [line.strip().split('\t') for line in raw_lines[header_idx+1:] if line.strip()]
    max_len = len(header)
    data = [d[:max_len] + ['']*(max_len-len(d)) for d in data]
    df = pd.DataFrame(data, columns=header)
    
    df.columns = [re.sub(r'\s*\[.*?\]', '', col).strip() for col in df.columns]
    speed_cols = [c for c in df.columns if 'm水平风速' in c and '最大' not in c]
    heights = sorted([int(re.search(r'(\d+)', c).group(1)) for c in speed_cols])
    
    # 构建矩阵
    n_samples = len(df)
    n_heights = len(heights)
    ws_mat = np.full((n_samples, n_heights), np.nan)
    ti_mat = np.full((n_samples, n_heights), np.nan)
    
    for i, h in enumerate(heights):
        ws_c = f'{h}m水平风速'
        std_c = next((c for c in df.columns if str(h) in c and ('偏差' in c or 'Std' in c) and '风向' not in c), None)
        if ws_c in df and std_c in df:
            w = pd.to_numeric(df[ws_c], errors='coerce').values
            s = pd.to_numeric(df[std_c], errors='coerce').values
            ws_mat[:, i] = w
            # 计算 TI，过滤低风速
            with np.errstate(divide='ignore', invalid='ignore'):
                ti = s / w
                ti[w < 3.0] = np.nan
                ti_mat[:, i] = ti

    # 提取 LLJ 事件
    events = []
    z_vec = np.array(heights)
    
    print(" -> 正在提取 LLJ 事件及其特征变量 (Z_jet, TI_base, TI_jet)...")
    
    for i in range(n_samples):
        ws = ws_mat[i, :]
        ti = ti_mat[i, :]
        if np.isnan(ws).any() or np.isnan(ti).any(): continue
        
        mx_i = np.argmax(ws)
        mx_h = heights[mx_i]
        
        # LLJ 判定
        if mx_h <= MIN_JET_HEIGHT or mx_h >= MAX_JET_HEIGHT: continue
        if (ws[mx_i] - ws[0] >= LLJ_THRESHOLD) and (ws[mx_i] - ws[-1] >= LLJ_THRESHOLD):
            # 提取关键变量
            z_jet = mx_h
            ti_base = ti[0]           # 假设最低高度为 Base
            ti_jet = ti[mx_i]         # 急流核心处的 TI
            
            # 只有当 TI_base > TI_jet 时，下层衰减模型才有意义
            if ti_base > ti_jet:
                events.append({
                    'ti_obs': ti,
                    'z_jet': z_jet,
                    'ti_base': ti_base,
                    'ti_jet': ti_jet,
                    'z_base': heights[0]
                })
                
    return z_vec, events

# --- 2. 定义智能模型 ---

def model_power_anchor(z, z_base, z_jet, ti_base, ti_jet, k_down, c_up, k_up):
    """
    模型 A: 幂律锚定
    下层: 使用归一化距离的幂律插值，强制连接 TI_base 和 TI_jet
    上层: 使用幂律增长
    """
    ti_pred = np.zeros_like(z, dtype=float)
    
    # Mask
    mask_lower = z <= z_jet
    mask_upper = z > z_jet
    
    # Lower Part: TI = TI_jet + (TI_base - TI_jet) * ((Z_jet - z)/(Z_jet - z_base))^k
    # 当 z=z_base -> term=1 -> TI=TI_base
    # 当 z=z_jet  -> term=0 -> TI=TI_jet
    if np.any(mask_lower):
        ratio = (z_jet - z[mask_lower]) / (z_jet - z_base)
        ratio = np.maximum(ratio, 0) # 保护
        ti_pred[mask_lower] = ti_jet + (ti_base - ti_jet) * np.power(ratio, k_down)
        
    # Upper Part: TI = TI_jet * (1 + C * (dist_from_jet)^k)
    if np.any(mask_upper):
        dist = z[mask_upper] - z_jet
        # 为了量纲统一，dist 除以 z_jet
        ti_pred[mask_upper] = ti_jet * (1 + c_up * np.power(dist / z_jet, k_up))
        
    return ti_pred

def model_gauss_anchor(z, z_base, z_jet, ti_base, ti_jet, gamma_down, c_up, k_up):
    """
    模型 B: 高斯锚定 (非对称高斯的泛化版)
    下层: 高斯衰减，但 Sigma 与 Z_jet 成正比 (Sigma = gamma * Z_jet)
    上层: 同模型 A
    """
    ti_pred = np.zeros_like(z, dtype=float)
    mask_lower = z <= z_jet
    mask_upper = z > z_jet
    
    # Lower Part: Asymmetric Gaussian Decay from Base
    # TI = TI_jet + (TI_base - TI_jet) * exp(...)
    # 注意：这里我们做一个近似，假设在 Base 处 exp 为 1 (z-z_base=0)，在 Jet 处 exp 衰减到很小
    # 但为了强制物理一致性，我们使用修正的高斯形式:
    if np.any(mask_lower):
        sigma = gamma_down * z_jet # Sigma 随急流高度动态变化
        term = np.exp(-((z[mask_lower] - z_base)**2) / (2 * sigma**2))
        # 此时 term 在 base=1. 在 jet 处不一定为0.
        # 为了强制通过 TI_jet, 我们需要一个归一化:
        val_at_jet = np.exp(-((z_jet - z_base)**2) / (2 * sigma**2))
        normalized_term = (term - val_at_jet) / (1 - val_at_jet)
        
        ti_pred[mask_lower] = ti_jet + (ti_base - ti_jet) * normalized_term

    # Upper Part
    if np.any(mask_upper):
        dist = z[mask_upper] - z_jet
        ti_pred[mask_upper] = ti_jet * (1 + c_up * np.power(dist / z_jet, k_up))
        
    return ti_pred

# --- 3. 全局训练 ---
def train_smart_models(z_vec, events):
    print(f"\n[训练中] 正在基于 {len(events)} 个事件进行全局参数寻优...")
    
    # 准备数据以加速
    # 转换为 numpy 结构可能太复杂，这里直接用循环累加 Loss，虽然慢点但逻辑清晰
    
    def global_loss_power(params):
        k_d, c_u, k_u = params
        total_sse = 0
        count = 0
        for ev in events:
            pred = model_power_anchor(z_vec, ev['z_base'], ev['z_jet'], ev['ti_base'], ev['ti_jet'], k_d, c_u, k_u)
            diff = ev['ti_obs'] - pred
            total_sse += np.sum(diff**2)
            count += len(diff)
        return total_sse / count # MSE

    def global_loss_gauss(params):
        g_d, c_u, k_u = params
        total_sse = 0
        count = 0
        for ev in events:
            pred = model_gauss_anchor(z_vec, ev['z_base'], ev['z_jet'], ev['ti_base'], ev['ti_jet'], g_d, c_u, k_u)
            diff = ev['ti_obs'] - pred
            total_sse += np.sum(diff**2)
            count += len(diff)
        return total_sse / count

    # 1. Train Power Anchor
    # k_down: 衰减形状 (0.5=convex, 1=linear, 2=concave)
    # c_up: 上升幅度系数
    # k_up: 上升形状 (1=linear, 2=quad)
    res_p = minimize(global_loss_power, [1.5, 2.0, 1.5], bounds=[(0.1, 5), (0, 10), (0.5, 3)], method='L-BFGS-B')
    
    # 2. Train Gauss Anchor
    # gamma_down: sigma/z_jet (0.1 ~ 1.0)
    res_g = minimize(global_loss_gauss, [0.3, 2.0, 1.5], bounds=[(0.05, 2), (0, 10), (0.5, 3)], method='L-BFGS-B')
    
    return res_p, res_g

# --- 4. 主程序 ---
def main():
    z_vec, events = load_data_matrix(FILE_PATH)
    if not events: return
    
    res_p, res_g = train_smart_models(z_vec, events)
    
    print("\n" + "="*60)
    print(" 🏆 智能锚定模型竞技结果 (Smart Anchor Arena)")
    print("="*60)
    print(f"Model A: Power Anchor MSE = {res_p.fun:.6f}")
    print(f"Model B: Gauss Anchor MSE = {res_g.fun:.6f}")
    
    best_model = "Power Anchor" if res_p.fun < res_g.fun else "Gauss Anchor"
    print(f"\n>>> 胜出者: 【{best_model}】")
    
    # 生成输出参数
    if best_model == "Power Anchor":
        k_d, c_u, k_u = res_p.x
        print("\n推荐通用公式参数 (固定系数):")
        print(f"  [下层衰减指数] k_down = {k_d:.4f}")
        print(f"  [上层回升系数] C_up   = {c_u:.4f}")
        print(f"  [上层回升指数] k_up   = {k_u:.4f}")
        
        print("\n>>> 最终公式 (将你的实测变量带入即可):")
        print("  1. 当 z <= Z_jet (下层):")
        print(f"     TI(z) = TI_jet + (TI_base - TI_jet) * [ (Z_jet - z) / (Z_jet - Z_base) ]^{k_d:.4f}")
        print("  2. 当 z > Z_jet (上层):")
        print(f"     TI(z) = TI_jet * [ 1 + {c_u:.4f} * ( (z - Z_jet) / Z_jet )^{k_u:.4f} ]")
        
    else:
        g_d, c_u, k_u = res_g.x
        print("\n推荐通用公式参数 (固定系数):")
        print(f"  [下层高斯因子] Gamma  = {g_d:.4f} (即 Sigma = {g_d:.2f} * Z_jet)")
        print(f"  [上层回升系数] C_up   = {c_u:.4f}")
        print(f"  [上层回升指数] k_up   = {k_u:.4f}")
        
        print("\n>>> 最终公式:")
        print("  1. 当 z <= Z_jet (下层):")
        print(f"     Sigma = {g_d:.4f} * Z_jet")
        print("     TI(z) = TI_jet + (TI_base - TI_jet) * Normalized_Gaussian(z, Sigma)")
        print("  2. 当 z > Z_jet (上层):")
        print(f"     TI(z) = TI_jet * [ 1 + {c_u:.4f} * ( (z - Z_jet) / Z_jet )^{k_u:.4f} ]")

    # 绘图验证 (取一个典型案例)
    # 找一个最接近平均 Z_jet 的事件来画图
    avg_zjet = np.mean([e['z_jet'] for e in events])
    sample_ev = min(events, key=lambda x: abs(x['z_jet'] - avg_zjet))
    
    plt.figure(figsize=(8, 10))
    plt.plot(sample_ev['ti_obs'], z_vec, 'ko', label='Observed (Sample Event)')
    
    # 预测
    pred_p = model_power_anchor(z_vec, sample_ev['z_base'], sample_ev['z_jet'], sample_ev['ti_base'], sample_ev['ti_jet'], *res_p.x)
    pred_g = model_gauss_anchor(z_vec, sample_ev['z_base'], sample_ev['z_jet'], sample_ev['ti_base'], sample_ev['ti_jet'], *res_g.x)
    
    plt.plot(pred_p, z_vec, 'b-', linewidth=2, label=f'Power Anchor (MSE={res_p.fun:.1e})')
    plt.plot(pred_g, z_vec, 'r--', linewidth=2, label=f'Gauss Anchor (MSE={res_g.fun:.1e})')
    
    # 标注变量
    plt.axhline(sample_ev['z_jet'], color='gray', linestyle=':', alpha=0.5)
    plt.text(np.min(sample_ev['ti_obs']), sample_ev['z_jet']+5, f"Z_jet={sample_ev['z_jet']}m", fontsize=10)
    
    plt.title(f'Smart Anchor Model Validation\nVariables: Z_jet, TI_base, TI_jet', fontsize=14)
    plt.xlabel('Turbulence Intensity')
    plt.ylabel('Height [m]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_img = os.path.join(OUTPUT_DIR, 'Smart_Anchor_Validation.png')
    plt.savefig(out_img, dpi=300)
    print(f"\n[图表] 验证图已保存: {out_img}")

if __name__ == "__main__":
    main()