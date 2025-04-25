import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 读取 'dataset/dataset_input.csv' 文件
input_file_path = './dataset/dataset_input.csv'
df = pd.read_csv(input_file_path)

# 不处理的列名
exclude_columns = ['DATA', 'DayOfYear_cos']
# 可以允许负值的列
allow_negative_column = 'dew_point_temperature'

# 只对这些列处理离群值

columns_to_handle_outliers_4q = ['KW', 'CHWTON', 'HTmmBTU', 'GHG']

columns_to_handle_outliers_quanju_3a = ['KW', 'CHWTON', 'GHG']

columns_to_handle_outliers_jubu_3a1 = [ 'HTmmBTU']

columns_to_handle_outliers_jubu_3a2 = ['CHWTON','KW', 'GHG']


# 定义负值和大正值处理函数，负值和大于300,000的值置为NaN
def handle_extreme_values(df, exclude_columns, allow_negative_column):
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in exclude_columns:
            # 处理负值，除了 dew_point_temperature 列外，其他列负值置为 NaN
            if col != allow_negative_column:
                df.loc[df[col] < 0, col] = np.nan
            # 处理超过 300,000 的大值
            df.loc[df[col] > 300000, col] = np.nan
    return df


# 定义全局四分位数法的异常值处理函数，仅对特定列进行
def handle_outliers_iqr(df, columns_to_handle_outliers):
    for col in columns_to_handle_outliers:
        # 计算Q1（25%分位数）和Q3（75%分位数）
        Q1 = df[col].quantile(0.15)
        Q3 = df[col].quantile(0.85)
        IQR = Q3 - Q1

        # 确定离群值范围
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 将离群值替换为NaN
        df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
    return df


# 定义基于3α原则的全局异常值处理函数，仅对特定列进行
def handle_outliers_3sigma(df, columns_to_handle_outliers):
    for col in columns_to_handle_outliers:
        mean = df[col].mean()  # 计算均值
        std = df[col].std()  # 计算标准差

        # 计算异常值的上限和下限
        upper_bound = mean + 3 * std
        lower_bound = mean - 3 * std

        # 将异常值替换为 NaN
        df.loc[(df[col] > upper_bound) | (df[col] < lower_bound), col] = np.nan
    return df


# 定义局部3α原则异常值处理函数，仅对特定列进行
def handle_local_outliers_3sigma(df, a, window_size=24, columns_to_handle_outliers=None):
    for col in columns_to_handle_outliers:
        # 使用滑动窗口对每个窗口内的数据进行3α法
        for i in range(len(df[col])):
            if pd.notna(df[col][i]):
                # 提取窗口内的数据
                window_data = df[col].iloc[max(0, i - window_size // 2):min(i + window_size // 2 + 1, len(df[col]))]
                mean = window_data.mean()
                std = window_data.std()

                # 计算局部范围
                upper_bound = mean + a * std
                lower_bound = mean - a * std

                # 将局部异常值替换为 NaN
                if df[col][i] > upper_bound or df[col][i] < lower_bound:
                    df[col][i] = np.nan
    return df


# 定义缺失值处理函数，使用线性插值，除了 exclude_columns 列
def fill_missing_values(df, exclude_columns=None):
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in exclude_columns:
            # 使用线性插值填充缺失值
            df[col] = df[col].interpolate(method='linear')
    return df


# 删除行范围的函数
def delete_rows(df, ranges):
    for start, end in sorted(ranges, reverse=True):
        df = df.drop(df.index[start:end + 1])  # 注意：end+1 确保删除到 end 行
    return df


# 指定要删除的行范围（按行号）
ranges_to_delete = [(1392, 2183), (9504, 10175), (20989, 21995)]

# 执行删除操作
df = delete_rows(df, ranges_to_delete)
# 重新索引删除行后的数据
df = df.reset_index(drop=True)

# 第一步：处理负值和大值，将负值（除 dew_point_temperature 外）和大于 300,000 的值置为 NaN
df = handle_extreme_values(df, exclude_columns, allow_negative_column)
df.to_csv("./dataset_final/dataset_bn.csv", index=False)

# 第二步：使用全局四分位数法去掉明显的异常值，仅对特定列处理
df = handle_outliers_iqr(df, columns_to_handle_outliers_4q)
df.to_csv("./dataset_final/dataset_quanju4Q.csv", index=False)

# # 第三步：进行全局异常值检测，使用3α原则，仅对特定列处理
df = handle_outliers_3sigma(df, columns_to_handle_outliers_quanju_3a)
df.to_csv("./dataset_final/dataset_quanju3α.csv", index=False)

# 第四步：进行局部异常值检测，使用局部3α原则，仅对特定列处理
df = handle_local_outliers_3sigma(df, window_size=24, columns_to_handle_outliers=columns_to_handle_outliers_jubu_3a1,
                                  a=2)
df = handle_local_outliers_3sigma(df, window_size=24, columns_to_handle_outliers=columns_to_handle_outliers_jubu_3a2,
                                  a=1.5)
output_file_path = './dataset_final/dataset_jubu3α.csv'
df.to_csv(output_file_path, index=False)

# 第五步：进行缺失值填充，使用线性插值，排除 DATA 和 DayOfYear_cos 列
df = fill_missing_values(df, exclude_columns=exclude_columns)

# 将处理后的数据保存到新的CSV文件
output_file_path = './dataset_final/dataset_input.csv'
df.to_csv(output_file_path, index=False)

print(f"数据处理完成：负值/大值处理、异常值检测（全局四分位法 + 3α原则）和缺失值填充已完成，结果已保存到 {output_file_path}")
