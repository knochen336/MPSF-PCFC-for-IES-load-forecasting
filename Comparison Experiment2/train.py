import argparse
import math
import time
import random
import torch
import torch.nn as nn
# from net import gtnet
from test_model.iTransformer import Model
import numpy as np
import importlib

from util import *
from trainer import Optim
import pandas as pd
from metrics import *
from ptflops import get_model_complexity_info
from net import gtnet
import os
from get_config import get_config
from Save_result_multipredict import show_pred_final
import warnings
warnings.filterwarnings("ignore")

test_model_name = 'iTransformer'

config = get_config(test_model_name)

pred_length = config.pred_len
seq_len = config.seq_len

save_name = f'{test_model_name}_{pred_length}_{seq_len}'

path1 = f'./output/{save_name}/result/'
if not os.path.exists(path1):
    os.makedirs(path1)
else:
    print('Exiting!')

path2 = f'./output/{save_name}/model/'
if not os.path.exists(path2):
    os.makedirs(path2)
else:
    print('Exiting!')

path3 = f'./output/{save_name}/assets/'
if not os.path.exists(path3):
    os.makedirs(path3)
else:
    print('Exiting!')


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        # [64,12,3]
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

    scale = data.scale.expand(predict.size(0), predict.size(1), 3).cpu().numpy()

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    mape = MAPE(predict * scale, Ytest * scale)
    mae = MAE(predict * scale, Ytest * scale)
    rmse = RMSE(predict * scale, Ytest * scale)

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return mae, mape, correlation, rmse


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    total_mae_loss = 0

    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        # print(X.shape, Y.shape)
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        # print(X.shape)
        tx = X
        ty = Y
        # print(tx.shape,ty.shape)
        output = model(tx)
        # print(output.shape)
        output = torch.squeeze(output)
        # print(output.shape)
        scale = data.scale.expand(output.size(0), output.size(1), 3)
        # print("ty", ty, "output", output, "scale", scale, "output*scale", output * scale, "ty*scale", ty * scale)
        # print(ty.shape,output.shape,scale.shape)
        loss = mape_loss(ty * scale, output * scale)
        loss_mae = MAE((ty * scale).cpu().detach().numpy(), (output * scale).cpu().detach().numpy())

        loss_mse = MSE(ty * scale, output * scale)
        loss.backward()
        total_loss += loss.item()
        total_mae_loss += loss_mae.item()
        grad_norm = optim.step()
        # if iter % 100 == 0:
        #     print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / 3))
        iter += 1

    return total_loss / iter, total_mae_loss / iter


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:

        _dict = {}
        for _, param in enumerate(model.named_parameters()):
            # print(param[0])
            # print(param[1])
            total_params = param[1].numel()
            # print(f'{total_params:,} total parameters.')
            k = param[0].split('.')[0]
            if k in _dict.keys():
                _dict[k] += total_params
            else:
                _dict[k] = 0
                _dict[k] += total_params
            # print('----------------')
        total_param = sum(p.numel() for p in model.parameters())
        bytes_per_param = 1
        total_bytes = total_param * bytes_per_param
        total_megabytes = total_bytes / (1024 * 1024)
        return total_param, total_megabytes, _dict


# 原始参数表：
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/dataset_input_jiuzheng.csv',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default=f'./output/{save_name}/model/model_lnn.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=15, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--in_dim', type=int, default=12, help='inputs dimension')

parser.add_argument('--seq_in_len', type=int, default=config.seq_len, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=config.pred_len, help='output sequence length')
parser.add_argument('--horizon', type=int, default=config.pred_len)

parser.add_argument('--layers', type=int, default=5, help='number of layers')

parser.add_argument('--batch_size', type=int, default=config.batchsize, help='batch size')
parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')

parser.add_argument('--clip', type=int, default=5, help='clip')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')

parser.add_argument('--epochs', type=int, default=config.epochs, help='')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

# 时序库的参数设置：
parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# forecasting task
parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=48,
                    help='the length of segmen-wise iteration of SegRNN')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility


def main():
    seed = 2020
    fix_seed(seed)

    fin = open(args.data)
    rawdat = np.loadtxt(fin, delimiter=',', skiprows=1)
    print(rawdat.shape)

    Data = DataLoaderS(args.data, 0.8, 0.1, device, args.horizon, args.seq_in_len, args.normalize)

    model = Model(config)
    model = model.to(device)

    flops, params = get_model_complexity_info(model, (1, 12, config.seq_len), as_strings=True, print_per_layer_stat=False)
    print('flops: ', flops, 'params: ', params)
    print('------------------------------------------------------')

    total_param, total_megabytes, _dict = count_parameters(model)
    # for k, v in _dict.items():
    #     print("Module:", k, "param:", v, "%3.3fM" % (v / (1024 * 1024)))
    print("Total megabytes:", total_megabytes, "M")
    print("Total parameters:", total_param)
    print(args)
    # print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    with open(f'./output/{save_name}/result/data.txt', 'a') as f:  # 设置文件对象
        # print('Number of model parameters is', nParams, flush=True, file=f)
        print('Parameters is', params, flush=True, file=f)
        print('FLOPs is', flops, flush=True, file=f)

    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    model = Model(config).to(device)

    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, config.lr, args.clip, 'min', config.weightdecay, config.decaypatience,
        lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            train_loss, train_mae_loss = train(Data, Data.train[0], Data.train[1][:, :, :3], model, criterion, optim,
                                               args.batch_size)
            val_mae, val_mape, val_corr, val_rmse = evaluate(Data, Data.valid[0], Data.valid[1][:, :, :3], model,
                                                             evaluateL2,
                                                             evaluateL1,
                                                             args.batch_size)
            optim.lronplateau(val_mape)
            with open(f'./output/{save_name}/result/data.txt', 'a') as f:  # 设置文件对象
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f} | train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f} | valid rmse  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, train_mae_loss, val_mae, val_mape,
                        val_corr, val_rmse), flush=True,
                    file=f)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f}| train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f} | learning rate  {:f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss,train_mae_loss, val_mae, val_mape, val_corr, optim.optimizer.param_groups[0]['lr']), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            # if val_mape < best_val:
            #     with open(args.save, 'wb') as f:
            #         torch.save(model, f)
            #     best_val = val_mape

            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_mape
            # print("model updated")
            if epoch % 1 == 0:
                test_mae, test_mape, test_corr, test_rmse = evaluate(Data, Data.test[0], Data.test[1][:, :, :3], model,
                                                                     evaluateL2,
                                                                     evaluateL1,
                                                                     args.batch_size)
                with open(f'./output/{save_name}/result/data.txt', 'a') as f:  # 设置文件对象
                    print(
                        "test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f}".format(test_mae,
                                                                                                              test_mape,
                                                                                                              test_corr,
                                                                                                              test_rmse),
                        flush=True, file=f)

                print("test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f}".format(test_mae,
                                                                                                            test_mape,
                                                                                                            test_corr,
                                                                                                            test_rmse),
                      flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    test_mae, test_mape, test_corr, test_rmse = evaluate(Data, Data.test[0], Data.test[1][:, :, :3], model, evaluateL2,
                                                         evaluateL1,
                                                         args.batch_size)
    with open(f'./output/{save_name}/result/data.txt', 'a') as f:  # 设置文件对象
        print("final test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f}".format(test_mae,
                                                                                                          test_mape,
                                                                                                          test_corr,
                                                                                                          test_rmse),
              file=f)
    print(
        "final test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f} | test rmse {:5.4f}".format(test_mae, test_mape,
                                                                                                    test_corr,
                                                                                                    test_rmse))

    all_y_true, all_predict_value = plow(Data, Data.test[0], Data.test[1][:, :, :3], model, args.batch_size)
    # 保存为 .pt 文件
    torch.save(all_y_true, f'./output/{save_name}/result/all_y_true.pt')
    torch.save(all_predict_value, f'./output/{save_name}/result/all_predict_value.pt')

    # 转换为 numpy 数组
    all_y_true_np = all_y_true.cpu().numpy()
    all_predict_value_np = all_predict_value.cpu().numpy()

    # 保存为 .npy 文件
    np.save(f'./output/{save_name}/result/all_y_true.npy', all_y_true_np)
    np.save(f'./output/{save_name}/result/all_predict_value.npy', all_predict_value_np)


    print('——————————————————————————————————final result——————————————————————————————')
    all_y_true_loaded_np = np.load(f'./output/{save_name}/result/all_y_true.npy')
    all_predict_value_loaded_np = np.load(f'./output/{save_name}/result/all_predict_value.npy')
    show_pred_final(all_y_true_loaded_np, all_predict_value_loaded_np, config.pred_len,save_name)


    return test_mae, test_mape, test_corr


def plow(data, X, Y, model, batch_size):
    model.eval()
    model.eval()

    all_predict_value = 0
    all_y_true = 0
    num = 0
    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        scale = data.scale.expand(output.size(0), output.size(1), 3)  # zuijin xiugai
        y_true = Y * scale
        predict_value = output * scale

        if num == 0:
            all_predict_value = predict_value
            all_y_true = y_true
        else:
            all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)
            all_y_true = torch.cat([all_y_true, y_true], dim=0)
        num = num + 1

    return all_y_true, all_predict_value


if __name__ == "__main__":
    main()
