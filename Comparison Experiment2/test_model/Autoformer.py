import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = 'long_term_forecast'    #configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.batch_size = configs.batch_size
        self.auxiliary = configs.auxiliary  # 获取辅助变量的数量

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        # 使用 series_decomp 进行时间序列的趋势和季节性分解。该部分用于长期预测

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # 对输入的时间序列数据进行嵌入，转换为 d_model 维度的特征

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Decoder  使用 DecoderLayer 解码器进行预测
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_dec（形状: [batch_size, pred_len, enc_in]）：解码器输入（预测目标的输入）

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # 对 x_enc 进行趋势和季节性分解，得到 trend_init 和 seasonal_init，这些用于初始化解码器的输入

        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # 都为[batch_size, label_len + pred_len, d_model]

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # 形状为 [batch_size, seq_len, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # [batch_size, seq_len, d_model]

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
    #    # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x_enc = x_enc.squeeze(1).permute(0, 2, 1)
        x_dec = torch.zeros_like(x_enc)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :3]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

#class Configs:
#    seq_len = 24  # 输入序列长度
#    label_len = 12  # 标签长度（即历史可用来预测的时间步数）
#    pred_len = 12  # 预测的时间步数
#    enc_in = 1  # 输入特征维度（假设只有一个目标变量）
#    dec_in = 1  # 解码器输入维度
#    d_model = 16  # 模型的隐藏层维度
#    batch_size = 32  # 批次大小
    #    embed = 'linear'  # 嵌入类型（线性）
    #    freq = 'h'  # 数据频率
    #    dropout = 0.1  # Dropout比例
    #    factor = 1  # 自注意力机制的因子
    #   output_attention = False  # 不输出注意力
    #   moving_avg = 25  # 趋势分解的滑动平均窗口
    #   e_layers = 2  # 编码器层数
    #  d_layers = 2  # 解码器层数
    #  n_heads = 4  # 多头注意力的头数
    #  d_ff = 128  # 前馈层的维度
    #  activation = 'gelu'  # 激活函数
    #  c_out = 1  # 输出通道数（预测一个目标变量）
    #   num_class = 10  # 分类任务的类别数（如果是分类任务的话）
    #   task_name = 'long_term_forecast'
#   auxiliary = 15

# 假设 Configs 类已经定义，并且包含上述的配置
#configs = Configs()
#model = Model(configs)

# 创建随机输入数据
#x_enc = torch.randn(32, 24, configs.enc_in)  # [batch_size, seq_len, enc_in]
#x_mark_enc = torch.randn(32, 24, configs.auxiliary)  # [batch_size, seq_len, 15]
#x_dec = torch.randn(32, 12, configs.enc_in)  # [batch_size, pred_len, enc_in]
# = torch.randn(32, 12, configs.auxiliary)  # [batch_size, pred_len, 15]

# 前向传播
#output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

# 输出结果形状
#print(output.shape)  # 应该输出 [32, 12, 1] 或其他任务的输出形状
