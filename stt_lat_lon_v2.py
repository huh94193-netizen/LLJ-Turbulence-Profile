import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

# --- 1. 数据准备 ---
data = {
    "Site": [
        "大庆萨尔图", "双鸭山集贤", "铁岭昌图", "铁岭台安", 
        "衡水故城", "潍坊昌邑", "开封杞县", "菏泽单县", 
        "盐城大丰", "孝感应城"
    ],
    # 修正后的经纬度
    "Longitude": [125.01, 130.94, 123.78, 122.56, 115.90, 119.45, 114.62, 116.10, 120.59, 113.61],
    "Latitude": [46.83, 46.94, 43.46, 41.51, 37.47, 37.07, 34.50, 34.71, 33.43, 31.00],
    "Region": ["东北", "东北", "东北", "东北", "华北", "华北", "华北", "华北", "华北", "华北"]
}
df = pd.DataFrame(data)

# --- 2. 画布设置 ---
# 设置中文字体 (根据您的系统调整，Windows通常是SimHei, Mac是Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 创建画布，指定投影 (PlateCarree是常用的经纬度等距投影)
plt.figure(figsize=(12, 10), dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())

# --- 3. 添加GIS地图要素 ---
# 设置地图显示范围 [经度小, 经度大, 纬度小, 纬度大]
# 根据您的场站位置，我们聚焦在中国东部和东北部，不显示西部空白区域以突出重点
extent = [110, 135, 28, 52] 
ax.set_extent(extent, crs=ccrs.PlateCarree())

# 添加地理特征
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5') # 陆地颜色
ax.add_feature(cfeature.OCEAN, facecolor='#cceeff') # 海洋颜色
ax.add_feature(cfeature.COASTLINE, linewidth=1)    # 海岸线
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5) # 国界
# 添加河流和湖泊增加细节感
ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.5)

# --- 4. 绘制场站 ---
colors = {'东北': '#d62728', '华北': '#1f77b4'} # 东北红，华北蓝

for region, group in df.groupby('Region'):
    # 绘制散点
    ax.scatter(group['Longitude'], group['Latitude'], 
               c=colors[region], label=region,
               s=80, alpha=0.9, edgecolors='white', linewidth=1,
               transform=ccrs.PlateCarree(), zorder=5)

# --- 5. 标注文本 ---
# 使用transform=ccrs.Geodetic()可以确保文本位置稍微智能一些（但这依然主要靠手动偏移）
for i, row in df.iterrows():
    # 给标签加一点偏移量，防止挡住点
    x_offset = 0.3
    y_offset = 0.3
    
    # 特殊处理：双鸭山太靠边，字往左放
    if "双鸭山" in row['Site']:
        x_offset = -2.5
        
    ax.text(row['Longitude'] + x_offset, row['Latitude'] + 0.1, row['Site'],
            transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold', zorder=6)

# --- 6. 装饰 ---
ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
plt.title('激光测风雷达场站地理分布 (GIS视图)', fontsize=16, pad=20)
plt.legend(loc='lower right', title="所属区域", fancybox=True, shadow=True)

# --- 7. 保存结果 ---
save_path = 'china_station_map_gis.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"地图已生成并保存至: {save_path}")

# 显示图片 (如果在服务器无界面环境，可注释此行)
# plt.show()