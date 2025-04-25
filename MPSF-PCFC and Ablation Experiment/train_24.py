import argparse
import math
import time
import random
import torch
import torch.nn as nn
# from net import gtnet
from torch_cfc_test_xtrunk2 import Cfc
import numpy as np
import importlib

from util import *
from trainer import Optim
import pandas as pd
from metrics import *


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
        if iter % 100 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / 3))
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
        bytes_per_param = 4
        total_bytes = total_param * bytes_per_param
        total_megabytes = total_bytes / (1024 * 1024)
        return total_param, total_megabytes, _dict


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/dataset_input_jiuzheng.csv',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model_lnn24.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda', help='')
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
parser.add_argument('--in_dim', type=int, default=24, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=168, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=24, help='output sequence length')
parser.add_argument('--horizon', type=int, default=24)
parser.add_argument('--layers', type=int, default=5, help='number of layers')

parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=0.008, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')

parser.add_argument('--clip', type=int, default=5, help='clip')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')

parser.add_argument('--epochs', type=int, default=40, help='')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)


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
    seed = 2024
    fix_seed(seed)

    fin = open(args.data)
    rawdat = np.loadtxt(fin, delimiter=',', skiprows=1)
    print(rawdat.shape)

    Data = DataLoaderS(args.data, 0.8, 0.1, device, args.horizon, args.seq_in_len, args.normalize)

    # model = gtnet(Data.train_feas, args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
    #               device, dropout=args.dropout, subgraph_size=args.subgraph_size,
    #               node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
    #               conv_channels=args.conv_channels, residual_channels=args.residual_channels,
    #               skip_channels=args.skip_channels, end_channels=args.end_channels,
    #               seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
    #               layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)

    # mask = torch.ones(64, 12, 168)
    # mask = torch.ones(args.batch_size, args.num_nodes, args.seq_in_len)

    hparams = {
        "backbone_activation": "lecun",  # 激活函数类型：可选值为 "silu", "relu", "tanh", "gelu", "lecun"
        "backbone_units": 64,  # backbone 网络的隐藏层单元数
        "backbone_layers": 2,  # backbone 网络的层数
        # "backbone_dr":0.2,
        "no_gate": False,  # 是否禁用门控机制，布尔值
        "minimal": False  # 是否使用最小化设置，布尔值
    }

    model = Cfc(batch=args.batch_size, seq_in_len=args.seq_in_len, seq_out_len=args.seq_out_len,
                in_features=args.num_nodes, hidden_size=64,
                out_feature=3, hparams=hparams)

    total_param, total_megabytes, _dict = count_parameters(model)
    model = model.to(device)
    for k, v in _dict.items():
        print("Module:", k, "param:", v, "%3.3fMB" % (v * 4 / (1024 * 1024)))
    print("Total megabytes:", total_megabytes, "MB")
    print("Total parameters:", total_param)
    print(args)
    # print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    with open('./result/data24.txt', 'a') as f:  # 设置文件对象
        print('Number of model parameters is', nParams, flush=True, file=f)

    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, 'min', 0.5, 3, lr_decay=args.weight_decay
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
            with open('./result/data24.txt', 'a') as f:  # 设置文件对象
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f} | train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f} | valid rmse  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, train_mae_loss, val_mae, val_mape,
                        val_corr, val_rmse), flush=True,
                    file=f)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f}| train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f} | valid rmse  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, train_mae_loss, val_mae, val_mape, val_corr,
                    val_rmse),
                flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            # if val_mape < best_val:
            #     with open(args.save, 'wb') as f:
            #         torch.save(model, f)
            #     best_val = val_mape

            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_mape
            print("model updated")
            if epoch % 1 == 0:
                test_mae, test_mape, test_corr, test_rmse = evaluate(Data, Data.test[0], Data.test[1][:, :, :3], model,
                                                                     evaluateL2,
                                                                     evaluateL1,
                                                                     args.batch_size)
                with open('./result/data24.txt', 'a') as f:  # 设置文件对象
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
    with open('./result/data24.txt', 'a') as f:  # 设置文件对象
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
    torch.save(all_y_true, './result/all_y_true24.pt')
    torch.save(all_predict_value, './result/all_predict_value24.pt')

    # 转换为 numpy 数组
    all_y_true_np = all_y_true.cpu().numpy()
    all_predict_value_np = all_predict_value.cpu().numpy()

    # 保存为 .npy 文件
    np.save('./result/all_y_true24.npy', all_y_true_np)
    np.save('./result/all_predict_value24.npy', all_predict_value_np)



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
