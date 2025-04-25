import os
import random
#0.008 0.5 3
import numpy as np
import torch
import torch.nn as nn
from FF import FF
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility
fix_seed(2024)
class SFF(nn.Module):
    def __init__(self, seq_len=168, pred_len=168, enc_in=12, period_len=24):
        super(SFF, self).__init__()

        # get parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period_len = period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        # seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        # x = (x - seq_mean).permute(0, 2, 1)
        x = x.permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        y = self.linear(x)  # bc,w,m

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1)

        return y



hparam = {
    "backbone_activation": "lecun",  # 激活函数类型：可选值为 "silu", "relu", "tanh", "gelu", "lecun"
    "backbone_units": 64,  # backbone 网络的隐藏层单元数
    "backbone_layers": 1,  # backbone 网络的层数
    # Dropout 概率（可选，若不需要则可以省略）
    "no_gate": False,  # 是否禁用门控机制，布尔值
    "minimal": False  # 是否使用最小化设置，布尔值
}


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfcCell(nn.Module):
    def __init__(self, input_size, hidden_size, hparams):
        super(CfcCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hparams = hparams
        self._no_gate = False
        if "no_gate" in self.hparams:
            self._no_gate = self.hparams["no_gate"]
        self._minimal = False
        if "minimal" in self.hparams:
            self._minimal = self.hparams["minimal"]

        if self.hparams["backbone_activation"] == "silu":
            backbone_activation = nn.SiLU
        elif self.hparams["backbone_activation"] == "relu":
            backbone_activation = nn.ReLU
        elif self.hparams["backbone_activation"] == "tanh":
            backbone_activation = nn.Tanh
        elif self.hparams["backbone_activation"] == "gelu":
            backbone_activation = nn.GELU
        elif self.hparams["backbone_activation"] == "lecun":
            backbone_activation = LeCun
        else:
            raise ValueError("Unknown activation")
        layer_list = [
            nn.Linear(input_size + hidden_size, self.hparams["backbone_units"]),
            backbone_activation(),
        ]
        for i in range(1, self.hparams["backbone_layers"]):
            layer_list.append(
                nn.Linear(
                    self.hparams["backbone_units"], self.hparams["backbone_units"]
                )
            )
            layer_list.append(backbone_activation())
            if "backbone_dr" in self.hparams.keys():
                layer_list.append(torch.nn.Dropout(self.hparams["backbone_dr"]))
        self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Linear(self.hparams["backbone_units"], hidden_size)
        if self._minimal:
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_size), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_size), requires_grad=True
            )
        else:
            self.ff2 = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_a = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_b = nn.Linear(self.hparams["backbone_units"], hidden_size)
        self.init_weights()

    def init_weights(self):
        init_gain = self.hparams.get("init")
        if init_gain is not None:
            for w in self.parameters():
                if w.dim() == 2:
                    torch.nn.init.xavier_uniform_(w, gain=init_gain)

    def forward(self, input, hx, ts):

        batch_size = input.size(0)
        if ts.size(0) != batch_size:
            # Adjust ts if necessary, e.g., repeat last value
            ts = ts[:batch_size]

        ts = ts.view(batch_size, 1)

        x = torch.cat([input, hx], 1)

        x = self.backbone(x)
        if self._minimal:
            # Solution
            ff1 = self.ff1(x)
            new_hidden = (
                    -self.A
                    * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                    * ff1
                    + self.A
            )
        else:
            # Cfc
            ff1 = self.tanh(self.ff1(x))
            ff2 = self.tanh(self.ff2(x))
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden







class Cfc(nn.Module):
    def __init__(
            self,
            batch,
            seq_in_len,
            seq_out_len,
            in_features,
            hidden_size,
            out_feature,
            hparams,
            return_sequences=True,
            chunk_length=6
    ):
        super(Cfc, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_feature
        self.return_sequences = return_sequences
        self.seq_in_len = seq_in_len
        self.seq_out_len = seq_out_len
        self.batch = batch
        self.use_mixed=False
        self.timespans = torch.full((self.batch, self.seq_in_len), 1.0)
        self.dropout = torch.nn.Dropout(0.15)
        self.rnn_cell = CfcCell(self.in_features, self.hidden_size, hparams)
        # 这里的全连接层用于将输出从 [64, 168, 12] 转换为 [64, 168, 3]
        self.fc = nn.Linear(self.hidden_size, self.out_features)

        self.tsf_period1_3 = SFF(seq_len=self.seq_in_len, period_len=3, pred_len=self.seq_in_len,
                                 enc_in=self.in_features)
        self.tsf_period1_6 = SFF(seq_len=self.seq_in_len, period_len=6, pred_len=self.seq_in_len,
                                 enc_in=self.in_features)
        self.tsf_period1_12 = SFF(seq_len=self.seq_in_len, period_len=12, pred_len=self.seq_in_len,
                                  enc_in=self.in_features)
        self.tsf_period1_24 = SFF(seq_len=self.seq_in_len, period_len=24, pred_len=self.seq_in_len,
                                  enc_in=self.in_features)

        self.tsf_period2_3 = SFF(seq_len=self.seq_in_len, period_len=3, pred_len=self.seq_out_len,
                                 enc_in=self.out_features)
        self.tsf_period2_6 = SFF(seq_len=self.seq_in_len, period_len=6, pred_len=self.seq_out_len,
                                 enc_in=self.out_features)
        self.tsf_period2_12 = SFF(seq_len=self.seq_in_len, period_len=12, pred_len=self.seq_out_len,
                                  enc_in=self.out_features)
        self.tsf_period2_24 = SFF(seq_len=self.seq_in_len, period_len=24, pred_len=self.seq_out_len,
                                  enc_in=self.out_features)

        self.tsf_period3_3 = SFF(seq_len=self.seq_in_len, period_len=3, pred_len=self.seq_in_len,
                                 enc_in=self.out_features)
        self.tsf_period3_6 = SFF(seq_len=self.seq_in_len, period_len=6, pred_len=self.seq_in_len,
                                 enc_in=self.out_features)
        self.tsf_period3_12 = SFF(seq_len=self.seq_in_len, period_len=12, pred_len=self.seq_in_len,
                                  enc_in=self.out_features)
        self.tsf_period3_24 = SFF(seq_len=self.seq_in_len, period_len=24, pred_len=self.seq_in_len,
                                  enc_in=self.out_features)

        self.fits_3 = FF(seq_len=self.seq_in_len, pred_len=self.seq_out_len, individual=True,
                         enc_in=self.out_features)
        self.chunk_length = chunk_length

    def forward(self, x, timespans=None, mask=None):
        x = x.squeeze(1)

        x = x.permute(0, 2, 1)
        seq_mean = torch.mean(x, dim=1).unsqueeze(2).permute(0, 2, 1)

        x = x - seq_mean

        x = (self.tsf_period1_3(x) + self.tsf_period1_6(x) + self.tsf_period1_12(x) + self.tsf_period1_24(
            x)) * 0.25

        x2 = (self.tsf_period2_3(x[:, :, :3]) + self.tsf_period2_6(x[:, :, :3]) + self.tsf_period2_12(
            x[:, :, :3]) + self.tsf_period2_24(x[:, :, :3])) * 0.25

        x3 = (self.tsf_period3_3(x[:, :, :3]) + self.tsf_period3_6(x[:, :, :3]) + self.tsf_period3_12(
            x[:, :, :3]) + self.tsf_period3_24(x[:, :, :3])) * 0.25

        seq_mean_out = seq_mean[:, :, 0:3]

        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        true_in_features = x.size(2)

        timespans = self.timespans.to(device)
        # mask = torch.ones(batch_size, seq_len).to(device)

        h_state = torch.zeros((batch_size, self.hidden_size), device=device)
        if self.use_mixed:
            c_state = torch.zeros((batch_size, self.hidden_size), device=device)
        output_sequence = []
        if mask is not None:
            forwarded_output = torch.zeros(
                (batch_size, self.out_feature), device=device
            )
            forwarded_input = torch.zeros((batch_size, true_in_features), device=device)
            time_since_update = torch.zeros(
                (batch_size, true_in_features), device=device
            )
        for t in range(seq_len):
            inputs = x[:, t]
            ts = timespans[:, t].squeeze()
            if mask is not None:
                if mask.size(-1) == true_in_features:
                    forwarded_input = (
                            mask[:, t] * inputs + (1 - mask[:, t]) * forwarded_input
                    )
                    time_since_update = (ts.view(batch_size, 1) + time_since_update) * (
                            1 - mask[:, t]
                    )
                else:
                    forwarded_input = inputs
                if (
                        true_in_features * 2 < self.in_features
                        and mask.size(-1) == true_in_features
                ):
                    # we have 3x in-features
                    inputs = torch.cat(
                        (forwarded_input, time_since_update, mask[:, t]), dim=1
                    )
                else:
                    # we have 2x in-feature
                    inputs = torch.cat((forwarded_input, mask[:, t]), dim=1)

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_state = self.rnn_cell.forward(inputs, h_state, ts)
            h_state = self.dropout(h_state)
            if mask is not None:
                cur_mask, _ = torch.max(mask[:, t], dim=1)
                cur_mask = cur_mask.view(batch_size, 1)
                current_output = self.fc(h_state)
                forwarded_output = (
                        cur_mask * current_output + (1.0 - cur_mask) * forwarded_output
                )
            if self.return_sequences:
                output_sequence.append(h_state)


        #readout = torch.cat(output_sequence, dim=1)
        readout = torch.stack(output_sequence, dim=1)

        readout = self.fc(readout)
        readout = readout + x3
        readout = self.fits_3(readout)
        readout = readout + seq_mean_out + x2

        return readout
