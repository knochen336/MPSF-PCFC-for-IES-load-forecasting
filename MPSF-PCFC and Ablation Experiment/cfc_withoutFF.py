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


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


import torch
import torch.nn as nn


class CfcCell(nn.Module):
    def __init__(self, input_size, hidden_size, hparams):
        super(CfcCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hparams = hparams
        self._no_gate = hparams.get("no_gate", False)
        self._minimal = hparams.get("minimal", False)

        if self.hparams["backbone_activation"] == "lecun":
            backbone_activation = LeCun
        else:
            backbone_activation = LeCun

        layer_list = [
            nn.Linear(input_size + hidden_size, self.hparams["backbone_units"]),
            backbone_activation(),
        ]
        for _ in range(1, self.hparams["backbone_layers"]):
            layer_list.append(
                nn.Linear(self.hparams["backbone_units"], self.hparams["backbone_units"])
            )
            layer_list.append(backbone_activation())
            if "backbone_dr" in self.hparams:
                layer_list.append(nn.Dropout(self.hparams["backbone_dr"]))

        self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Linear(self.hparams["backbone_units"], hidden_size)

        if self._minimal:
            self.w_tau = nn.Parameter(data=torch.zeros(1, hidden_size), requires_grad=True)
            self.A = nn.Parameter(data=torch.ones(1, hidden_size), requires_grad=True)
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
                    nn.init.xavier_uniform_(w, gain=init_gain)

    def forward(self, input, hx, ts):
        """
        input: [batch, chunk_length, input_size]
        hx: [batch, chunk_length, hidden_size] - initial hidden state for the chunk
        ts: [batch, chunk_length] - timespans for each time step in the chunk
        """
        batch_size, chunk_length, _ = input.size()
        if ts.size(0) != batch_size:
            # Adjust ts if necessary, e.g., repeat last value
            ts = ts[:batch_size, :]

        # Concatenate input and hx along the feature dimension
        x = torch.cat([input, hx], dim=2)  # [batch, chunk_length, input_size + hidden_size]

        # Apply backbone in parallel across the chunk length
        x = self.backbone(x)  # [batch, chunk_length, backbone_units]

        if self._minimal:
            # Minimal computation using tensor operations
            ff1 = self.ff1(x)  # [batch, chunk_length, hidden_size]
            w_tau_abs = torch.abs(self.w_tau).unsqueeze(0).unsqueeze(1)  # [1, 1, hidden_size]
            A_expanded = self.A.unsqueeze(0).unsqueeze(1)  # [1, 1, hidden_size]
            ts_expanded = ts.unsqueeze(2)  # [batch, chunk_length, 1]
            new_hidden = (
                    -A_expanded * torch.exp(-ts_expanded * (w_tau_abs + torch.abs(ff1))) * ff1 + A_expanded
            )  # [batch, chunk_length, hidden_size]
        else:
            # General computation using time_a, time_b, and gating mechanism
            ff1 = self.tanh(self.ff1(x))  # [batch, chunk_length, hidden_size]
            ff2 = self.tanh(self.ff2(x))  # [batch, chunk_length, hidden_size]
            t_a = self.time_a(x)  # [batch, chunk_length, hidden_size]
            t_b = self.time_b(x)  # [batch, chunk_length, hidden_size]
            t_interp = self.sigmoid(t_a * ts.unsqueeze(2) + t_b)  # [batch, chunk_length, hidden_size]

            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2  # [batch, chunk_length, hidden_size]
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2  # [batch, chunk_length, hidden_size]

        return new_hidden  # Return the new hidden state, shape [batch, chunk_length, hidden_size]


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

        self.fc2= nn.Linear(self.seq_in_len, self.seq_out_len)
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
        batch_size, seq_len, _ = x.size()
        timespans = self.timespans.to(device)

        # Initialize hidden state with zeroes, having the same chunk_length as used in input
        num_chunks_per_day = 24 // self.chunk_length  # Number of chunks in a single day
        h_state = torch.zeros((batch_size, self.chunk_length, self.hidden_size), device=device)
        h_state_last_day = [torch.zeros((batch_size, self.chunk_length, self.hidden_size), device=device)
                            for _ in range(num_chunks_per_day)]
        h_state_last_2_days = [torch.zeros((batch_size, self.chunk_length, self.hidden_size), device=device)
                               for _ in range(num_chunks_per_day)]
        h_state_last_3_days = [torch.zeros((batch_size, self.chunk_length, self.hidden_size), device=device)
                               for _ in range(num_chunks_per_day)]

        output_sequence = []

        # Process sequence in chunks
        for i in range(0, seq_len, self.chunk_length):
            chunk_end = min(i + self.chunk_length, seq_len)
            current_chunk_length = chunk_end - i

            x_chunk = x[:, i:chunk_end, :]  # [batch, chunk_length, in_features]
            ts_chunk = timespans[:, i:chunk_end]  # [batch, chunk_length]

            # Adjust h_state if last chunk is smaller than the defined chunk_length
            if current_chunk_length != self.chunk_length:
                h_state = h_state[:, :current_chunk_length, :]
                for j in range(num_chunks_per_day):
                    h_state_last_day[j] = h_state_last_day[j][:, :current_chunk_length, :]
                    h_state_last_2_days[j] = h_state_last_2_days[j][:, :current_chunk_length, :]
                    h_state_last_3_days[j] = h_state_last_3_days[j][:, :current_chunk_length, :]

            # Calculate the day chunk index
            day_chunk_index = (i // self.chunk_length) % num_chunks_per_day

            # Incorporate hidden states from last days
            h_state = (
                    h_state
                    + h_state_last_day[day_chunk_index]
                    + h_state_last_2_days[day_chunk_index] * 0.5
                    + h_state_last_3_days[day_chunk_index] * 0.3
            )

            # Forward through the RNN cell, get the new hidden state
            h_state = self.rnn_cell(x_chunk, h_state, ts_chunk)  # [batch, current_chunk_length, hidden_size]
            h_state = self.dropout(h_state)  # Apply dropout

            # Store the output sequence
            output_sequence.append(h_state)

            # Update the jump-propagated states for the next iterations
            if i + self.chunk_length < seq_len:  # Ensure we don't exceed seq_len
                last_day_chunk_index = (day_chunk_index + 1) % num_chunks_per_day
                last_2_days_chunk_index = (day_chunk_index + 2) % num_chunks_per_day
                last_3_days_chunk_index = (day_chunk_index + 3) % num_chunks_per_day

                # Update the hidden states for future days
                h_state_last_day[last_day_chunk_index] = h_state.clone()
                h_state_last_2_days[last_2_days_chunk_index] = h_state_last_day[last_day_chunk_index].clone()
                h_state_last_3_days[last_3_days_chunk_index] = h_state_last_2_days[last_2_days_chunk_index].clone()

        readout = torch.cat(output_sequence, dim=1)
        readout = self.fc(readout)
        readout = readout + x3
        readout=readout.permute(0,2,1)
        readout = self.fc2(readout)
        readout = readout.permute(0, 2, 1)
        readout = readout + seq_mean_out + x2

        return readout
