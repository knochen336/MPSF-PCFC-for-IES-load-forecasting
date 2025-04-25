import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
file_path = 'data_filtered/GHCNh_USW00023183_filtered_2022.csv'
df = pd.read_csv(file_path, dtype={'DATE': str}, low_memory=False)

# 保留感兴趣的列
columns_of_interest = ['temperature', 'dew_point_temperature', 'station_level_pressure', 'sea_level_pressure',
                       'wind_direction', 'wind_speed', 'relative_humidity', 'wet_bulb_temperature',
                       'visibility', 'altimeter', 'sky_cover_1', 'sky_cover_baseht_1',
                       'KW', 'CHWTON', 'HTmmBTU', 'Combined mmBTU', 'GHG', 'KWS']

# 过滤数据，保留相关列
df_filtered = df[columns_of_interest]

# 对于关键列进行缺失值填充（前后平均插值法）
for col in ['KW', 'CHWTON', 'HTmmBTU', 'Combined mmBTU', 'GHG', 'KWS']:
    df_filtered[col] = df_filtered[col].interpolate(method='linear', limit_direction='both')

# 定义处理离群值的函数，使用IQR（四分位距）方法
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# 处理关键列的离群值
columns_to_clean = ['KW', 'CHWTON', 'HTmmBTU', 'Combined mmBTU', 'GHG', 'KWS']
df_filtered_cleaned = remove_outliers(df_filtered, columns_to_clean)

# 计算 KW、CHWTON、HTmmBTU、Combined mmBTU、GHG、KWS 与其他列的相关性矩阵
correlation_matrix = df_filtered_cleaned.corr()

# 提取 KW, CHWTON, HTmmBTU, Combined mmBTU, GHG, KWS 相关的行和列
selected_columns = ['KW', 'CHWTON', 'HTmmBTU', 'Combined mmBTU', 'GHG', 'KWS']
correlation_matrix_result = correlation_matrix.loc[:, selected_columns]

# 打印相关性矩阵
print(correlation_matrix_result)

# 保存相关性矩阵到CSV文件
output_csv_path = 'result/correlation_matrix_result.csv'
correlation_matrix_result.to_csv(output_csv_path)
print(f"相关性矩阵已保存到 {output_csv_path}")

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_result, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")

# 设置标题和轴标签
plt.title('Correlation Heatmap: KW, CHWTON, HTmmBTU, Combined mmBTU, GHG, and KWS')
plt.xlabel('Target Columns (KW, CHWTON, HTmmBTU, Combined mmBTU, GHG, KWS)')
plt.ylabel('Feature Columns')

# 保存热力图到PNG文件
output_image_path = 'result/correlation_heatmap.png'
plt.savefig(output_image_path)
print(f"热力图已保存到 {output_image_path}")

# 显示热力图
plt.show()
