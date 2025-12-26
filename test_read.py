import pandas as pd

# 文件路径
file_path = '/home/huxun/02_LLJ/exported_data/大庆萨尔图-1445#-20240502-20251222-filter-Exported.txt'

# 跳过开头的注释行（前12行是说明，第13行是表头）
df = pd.read_csv(
    file_path,
    skiprows=12,  # 跳过前12行注释
    sep='\s+',    # 用空格分隔列
    engine='python'
)

# 查看数据结构
print("数据形状：", df.shape)
print("\n前5行数据：")
print(df.head())

# 查看列名（变量）
print("\n变量列表：")
print(df.columns.tolist())