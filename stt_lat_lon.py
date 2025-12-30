import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体 (根据环境调整，这里尽量使用通用设置或英文代替以防乱码，但在最终输出中会尝试显示中文)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 数据录入 (已修正经纬度列名反转的问题)
data = {
    "Site": [
        "大庆萨尔图", "双鸭山集贤", "铁岭昌图", "铁岭台安", 
        "衡水故城", "潍坊昌邑", "开封杞县", "菏泽单县", 
        "盐城大丰", "孝感应城"
    ],
    "Longitude": [125.01, 130.94, 123.78, 122.56, 115.90, 119.45, 114.62, 116.10, 120.59, 113.61],
    "Latitude": [46.83, 46.94, 43.46, 41.51, 37.47, 37.07, 34.50, 34.71, 33.43, 31.00],
    "Region": ["东北", "东北", "东北", "东北", "华北", "华北", "华北", "华北", "华北", "华北"]
}

df = pd.DataFrame(data)

# 创建画布
plt.figure(figsize=(10, 8))

# 绘制散点
# 东北区域用蓝色，华北区域用红色
colors = {'东北': 'tab:blue', '华北': 'tab:red'}
for region, group in df.groupby('Region'):
    plt.scatter(group['Longitude'], group['Latitude'], label=region, s=100, c=colors[region], edgecolors='k', alpha=0.8)

# 添加标签
for i, txt in enumerate(df['Site']):
    plt.annotate(txt, (df['Longitude'][i], df['Latitude'][i]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

# 设置坐标轴范围 (稍微扩大一点以便看清相对位置)
plt.xlim(110, 135)
plt.ylim(28, 50)

# 添加装饰
plt.title('激光测风雷达场站地理分布示意图', fontsize=15)
plt.xlabel('经度 (Longitude)', fontsize=12)
plt.ylabel('纬度 (Latitude)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="区域", loc='lower left')

# 添加简单的背景示意（非精确地图，仅辅助定位）
plt.axhline(y=40, color='gray', linestyle=':', alpha=0.3) # 北纬40度线
plt.text(111, 40.2, "北纬40°", color='gray', fontsize=8)

plt.tight_layout()

# plt.show()  # 注释掉这行
# 将图片保存到当前脚本运行的目录下，文件名为 map_distribution.png
plt.savefig('/home/huxun/02_LLJ/result/map_distribution.png', dpi=300) 
print("图片已保存至当前路径下的 map_distribution.png")