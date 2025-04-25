import pandas as pd

# 处理气象数据
qixiang_file_path = 'data_filtered/dataset_qixiang.csv'
df_qixiang = pd.read_csv(qixiang_file_path)

# 将 'DATE' 列转换为 datetime 类型，并将时间加上 9 分钟
df_qixiang['DATE'] = pd.to_datetime(df_qixiang['DATE']) + pd.Timedelta(minutes=9)

# 创建完整的日期范围（从最小到最大日期，以1小时为步长）
full_date_range = pd.date_range(start=df_qixiang['DATE'].min(), end=df_qixiang['DATE'].max(), freq='H')

# 创建一个完整日期的 DataFrame
full_df = pd.DataFrame({'DATE': full_date_range})

# 将原始数据与完整的日期范围合并，确保所有日期都存在
df_qixiang_merged = pd.merge(full_df, df_qixiang, on='DATE', how='left')

# 保存合并后的气象数据
df_qixiang_merged.to_csv('dataset/dataset_LR.csv', index=False)
print("处理后的气象数据已保存到 'dataset/dataset_LR.csv'")

# 处理负荷数据
load_file_path = 'data_filtered/dataset_load.csv'
df_load = pd.read_csv(load_file_path)

# 根据 'Year', 'Month', 'Day', 'Hour' 创建时间戳并转换为 datetime 类型
df_load['DATE'] = pd.to_datetime(df_load[['Year', 'Month', 'Day', 'Hour']])

# 只保留负荷相关列
df_load = df_load[['DATE', 'KW', 'KWS', 'CHWTON', 'HTmmBTU', 'Combined mmBTU', 'GHG', 'DOW']]

# 读取合并后的气象数据
df_merged = pd.read_csv('dataset/dataset_LR.csv')

# 确保 'DATE' 列类型一致，转换为 datetime 类型
df_merged['DATE'] = pd.to_datetime(df_merged['DATE'])

# 将负荷数据合并到气象数据中
df_final = pd.merge(df_merged, df_load, on='DATE', how='left')

# 保存最终数据到 CSV 文件
output_final_path = 'dataset/dataset_final.csv'
df_final.to_csv(output_final_path, index=False)
print(f"最终处理后的数据已保存到 {output_final_path}")
