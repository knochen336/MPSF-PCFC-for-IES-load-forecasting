# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 读取 'dataset_final/dataset_input.csv' 文件
# file_path = 'dataset_final/dataset_input.csv'
# df = pd.read_csv(file_path)
#
# # 目标列
# target_columns = ['KW', 'CHWTON', 'HTmmBTU']
#
# # 排除的列
# exclude_columns = ['DATE']
#
# # 只保留参与计算的列（剔除 DATE 列）
# df_corr = df.drop(columns=exclude_columns)
#
# # 定义相关性计算方法
# correlation_methods = ['pearson', 'spearman']
#
# # 计算并绘制每种相关性方法的热力图
# for method in correlation_methods:
# 		# 计算目标列与其他所有列的相关性
# 		corr_matrix = df_corr.corr(method=method)[target_columns]
#
# 		# 绘制热力图
# 		plt.figure(figsize=(10, 8))
# 		sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
#
# 		plt.title(f'{method.capitalize()} Correlation Heatmap for KW, CHWTON, HTmmBTU, GHG with Other Metrics')
# 		plt.tight_layout()
#
# 		# 保存热力图为 PNG 文件
# 		#output_image_path = f'dataset_final/correlation_heatmap_{method}.png'
# 		#plt.savefig(output_image_path)
# 		plt.show()
#
# 		#print(f'{method.capitalize()} 相关性热力图已保存为 {output_image_path}')








# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import textwrap
#
# # 读取数据
# df = pd.read_csv('dataset_final/dataset_input.csv')
#
# # 定义目标列和排除列
# target_columns = ['KW', 'CHWTON', 'HTmmBTU']
# exclude_columns = ['DATE']
#
# # 处理数据
# df_corr = df.drop(columns=exclude_columns)
#
#
# # 定义自定义行名换行函数
# def wrap_labels(labels, max_length=16):
# 	return [textwrap.fill(text.replace(' ', '\n'), width=max_length)
# 			if len(text) > max_length else text
# 			for text in labels]
#
#
# # 获取原始行名列表
# original_labels = ['Temperature', 'Dew Point Temperature', 'Station Level Pressure',
# 				   'Sea Level Pressure', 'Wet Bulb Temperature', 'Altimeter', 'Day Of Year',
# 				   'Combined mmBTU', 'KW', 'CHWTON', 'HTmmBTU', 'GHG']
#
# # 创建带换行的标签（手动调整关键长标签）
# wrapped_labels = [
# 	'Temperature',
# 	'Dew Point\nTemperature',
# 	'Station Level\nPressure',
# 	'Sea Level\nPressure',
# 	'Wet Bulb\nTemperature',
# 	'Altimeter',
# 	'Day Of Year',
# 	'Combined\nmmBTU',
# 	'KW',
# 	'CHWTON',
# 	'HTmmBTU',
# 	'GHG'
# ]
#
# # 设置全局字体参数
# plt.rcParams.update({
# 	'axes.titlesize': 16,  # 主标题大小
# 	'axes.labelsize': 14,  # 轴标签大小
# 	'xtick.labelsize': 12,  # X轴刻度
# 	'ytick.labelsize': 12,  # Y轴刻度
# 	'figure.titlesize': 18  # 图形标题（如果使用）
# })
#
# for method in ['pearson', 'spearman']:
# 	# 计算相关性矩阵
# 	corr_matrix = df_corr.corr(method=method)[target_columns]
#
# 	# 创建图形（调整宽高比例）
# 	fig, ax = plt.subplots(figsize=(7, 9))  # 增加宽度以适应换行
#
# 	# 绘制热力图（调整单元格比例）
# 	heatmap = sns.heatmap(
# 		corr_matrix,
# 		annot=True,
# 		annot_kws={'size': 13},  # 相关性数字大小
# 		cmap='coolwarm',
# 		fmt=".2f",
# 		vmin=-1,
# 		vmax=1,
# 		square=False,  # 关闭正方形单元格
# 		linewidths=0.5,
# 		cbar_kws={
# 			'shrink': 0.9,  # 缩短颜色条长度
# 			'label': 'Correlation'
# 		}
# 	)
#
# 	# 设置自定义Y轴标签
# 	ax.set_yticklabels(
# 		wrapped_labels,
# 		rotation=0,  # 水平显示
# 		va='center'  # 垂直居中
# 	)
#
# 	# 调整X轴标签
# 	ax.set_xticklabels(
# 		ax.get_xticklabels(),
#
# 	)
#
# 	# 设置标题和布局
# 	plt.title(
# 		f'{method.capitalize()} Correlation',
# 		pad=20,  # 增加标题与图形的间距
# 		fontsize=16
# 	)
#
# 	# 调整颜色条标签大小
# 	cbar = heatmap.collections[0].colorbar
# 	cbar.ax.tick_params(labelsize=12)
# 	cbar.set_label('Correlation Coefficient', size=12)
#
# 	# 优化布局
# 	plt.tight_layout()
#
# 	# 显示图形
# 	plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

# 读取数据
df = pd.read_csv('dataset_final/dataset_input.csv')

# 定义目标列和排除列
target_columns = ['KW', 'CHWTON', 'HTmmBTU']
exclude_columns = ['DATE']

# 处理数据
df_corr = df.drop(columns=exclude_columns)

# 创建带换行的标签
wrapped_labels = [
    'Temperature',
    'Dew Point\nTemperature',
    'Station Level\nPressure',
    'Sea Level\nPressure',
    'Wet Bulb\nTemperature',
    'Altimeter',
    'Day Of Year',
    'Combined\nmmBTU',
    'KW',
    'CHWTON',
    'HTmmBTU',
    'GHG'
]

# 保持原有全局字体设置
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 14.5,
    'figure.titlesize': 20
})

# 创建1x2子图布局，不共享y轴
fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=False)
# 微调子图间距
plt.subplots_adjust(wspace=0.05, left=0.1, right=0.88, top=0.9, bottom=0.1)

# 存储热力图对象用于颜色条
heatmap = None

for i, method in enumerate(['pearson', 'spearman']):
    ax = axes[i]
    corr_matrix = df_corr.corr(method=method)[target_columns]

    # 绘制热力图（严格保持原有参数）
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        annot_kws={'size': 16},
        cmap='coolwarm',
        fmt=".2f",
        vmin=-1,
        vmax=1,
        square=False,
        linewidths=0.5,
        cbar=False,
        ax=ax
    )

    # 只为左侧子图设置 Y 轴标签
    if i == 0:
        # 调整 y 轴刻度位置，以便它们准确对齐到每个格子的中心
        ax.set_yticks(range(len(wrapped_labels)))  # 设置 y 轴刻度的位置
        ax.set_yticklabels(
            wrapped_labels[:len(corr_matrix)],  # 确保标签长度匹配
            rotation=0,
            va='center',  # 垂直居中
            ha='right'  # 标签右对齐
        )
    else:
        ax.set_yticklabels([])  # 右侧不显示标签

    # 设置 X 轴标签（保持原样）
    ax.set_xticklabels(
        ax.get_xticklabels()  # 保持原有显示方式
    )

    # 设置标题
    ax.set_title(f'{method.capitalize()} Correlation', pad=20)

# 添加全局颜色条（严格保持原有格式）
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.68])  # 精确位置控制
fig.colorbar(heatmap.collections[0], cax=cbar_ax, label='Correlation Coefficient')
cbar_ax.tick_params(labelsize=15)
cbar_ax.yaxis.label.set_size(17)
#plt.show()
# 显示图像
plt.savefig('correlation_matrix.pdf')




