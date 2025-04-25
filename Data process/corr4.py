import pandas as pd
import numpy as np

# 读取CSV文件
file_path = 'dataset/dataset_final.csv'
df = pd.read_csv(file_path)

# 将 'DATE' 列转换为 datetime 类型
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y/%m/%d %H:%M')

# 提取 'Year', 'Month', 'Day', 'Hour' 从 'DATE' 列
df['Year'] = df['DATE'].dt.year
df['Month'] = df['DATE'].dt.month
df['Day'] = df['DATE'].dt.day
df['Hour'] = df['DATE'].dt.hour

# 创建 'DayOfYear' 和 'HourOfDay' 列
df['DayOfYear'] = df['DATE'].dt.dayofyear
df['HourOfDay'] = df['Hour']

# 将日期信息转化为周期性特征
df['DayOfYear_cos'] = (np.cos(2 * np.pi * df['DayOfYear'] / 365) + 1) / 2

# 保留感兴趣的列
columns_to_keep = [
    'DATE',
    'temperature',
    'dew_point_temperature',
    'station_level_pressure',
    'sea_level_pressure',
    'wet_bulb_temperature',
    'altimeter',
    'DayOfYear_cos',
    'Combined mmBTU',
    'KW',
    'CHWTON',
    'HTmmBTU',
    'GHG'
]

# 选择要保留的列
df_filtered = df[columns_to_keep].copy()

# 将结果保存到新的CSV文件
output_file_path = 'dataset/dataset_input.csv'
df_filtered.to_csv(output_file_path, index=False)
print(f"数据已保存到 {output_file_path}")

