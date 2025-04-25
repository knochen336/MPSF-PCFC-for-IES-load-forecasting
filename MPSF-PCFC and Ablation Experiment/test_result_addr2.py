import os

import numpy as np
import matplotlib.pyplot as plt
import math
from metrics import MAE, MAPE, RMSE
import csv

from sklearn.metrics import r2_score


def show_pred(all_y_true, all_predict_values):
    true_for_r2 = all_y_true.reshape(-1, 3)
    predict_for_r2 = all_predict_values.reshape(-1, 3)

    mae1 = MAE(all_y_true[:, :, :1], all_predict_values[:, :, :1])
    mape1 = MAPE(all_y_true[:, :, :1], all_predict_values[:, :, :1])
    rmase1 = RMSE(all_y_true[:, :, :1], all_predict_values[:, :, :1])
    r2_1 = r2_score(true_for_r2[:, 0], predict_for_r2[:, 0])
    print("============整个测试集电负荷===================")
    print("MAE = " + str(mae1))
    print("MAPE = " + str(mape1))
    print("RMSE = " + str(rmase1))
    print("R2 = " + str(r2_1))

    print("============整个测试集cooling==================")
    mae2 = MAE(all_y_true[:, :, 1:2], all_predict_values[:, :, 1:2])
    mape2 = MAPE(all_y_true[:, :, 1:2], all_predict_values[:, :, 1:2])
    rmase2 = RMSE(all_y_true[:, :, 1:2], all_predict_values[:, :, 1:2])
    r2_2 = r2_score(true_for_r2[:, 1], predict_for_r2[:, 1])
    print("MAE = " + str(mae2))
    print("MAPE = " + str(mape2))
    print("RMSE = " + str(rmase2))
    print("R2 = " + str(r2_2))

    print("==============整个测试集heating=================")
    mae3 = MAE(all_y_true[:, :, 2:3], all_predict_values[:, :, 2:3])
    mape3 = MAPE(all_y_true[:, :, 2:3], all_predict_values[:, :, 2:3])
    rmase3 = RMSE(all_y_true[:, :, 2:3], all_predict_values[:, :, 2:3])
    r2_3 = r2_score(true_for_r2[:, 2], predict_for_r2[:, 2])

    print("MAE = " + str(mae3))
    print("MAPE = " + str(mape3))
    print("RMSE = " + str(rmase3))
    print("R2 = " + str(r2_3))

    mae = MAE(all_y_true, all_predict_values)
    rmse = RMSE(all_y_true, all_predict_values)
    mape = MAPE(all_y_true, all_predict_values)
    r2 = r2_score(true_for_r2, predict_for_r2)

    print("ST-GCN all mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}, r2: {:02.4f}".format(mae, rmse, mape, r2))


def main(directory="./result/"):
    # 收集所有预测和真实值文件
    predict_files = {}
    true_files = {}

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            # 提取模型名
            if filename.startswith("all_predict_value"):
                model_name = filename[len("all_predict_value"):-4]  # 去掉前缀和后缀
                predict_files[model_name] = os.path.join(directory, filename)
            elif filename.startswith("all_y_true"):
                model_name = filename[len("all_y_true"):-4]  # 去掉前缀和后缀
                true_files[model_name] = os.path.join(directory, filename)

    # 找出所有匹配的模型
    common_models = set(predict_files.keys()) & set(true_files.keys())
    if not common_models:
        print("错误：未找到匹配的预测值和真实值文件！")
        return

    # 按模型名排序后处理
    for model_name in sorted(common_models):
        print(f"\n\n{'=' * 50}")
        print(f"===========================正在处理模型: {model_name}=================================")
        print(f"{'=' * 50}\n")

        # 加载数据
        try:
            y_true = np.load(true_files[model_name])
            y_pred = np.load(predict_files[model_name])
        except Exception as e:
            print(f"加载文件失败: {e}")
            continue

        # 检查形状一致性
        if y_true.shape != y_pred.shape:
            print(f"形状不匹配！真实值形状: {y_true.shape}, 预测值形状: {y_pred.shape}")
            continue

        # 执行评估
        show_pred(y_true, y_pred)

if __name__ == "__main__":
    main()
