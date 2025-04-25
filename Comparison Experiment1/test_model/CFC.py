import numpy as np
import torch
import torch.nn as nn

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


class Model(nn.Module):
    def __init__(
            self,
            config,
            return_sequences=True,
            use_ltc=False,
    ):
        super(Model, self).__init__()
        self.in_features = config.in_features
        self.hidden_size = config.hidden_size
        self.out_feature = config.out_feature
        self.return_sequences = return_sequences
        self.seq_in_len = config.seq_in_len
        self.seq_out_len = config.seq_out_len
        self.use_mixed = None

        self.batch = config.batch
        self.timespans = torch.full((self.batch, self.seq_in_len), 1.0)
        self.dropout = torch.nn.Dropout(0.3)
        if use_ltc:
            self.rnn_cell = LTCCell(self.in_features, self.hidden_size)
        else:
            hparams = {
                "backbone_activation": "lecun",  # 激活函数类型：可选值为 "silu", "relu", "tanh", "gelu", "lecun"
                "backbone_units": 64,  # backbone 网络的隐藏层单元数
                "backbone_layers": 1,  # backbone 网络的层数
                # Dropout 概率（可选，若不需要则可以省略）
                "no_gate": False,  # 是否禁用门控机制，布尔值
                "minimal": False  # 是否使用最小化设置，布尔值
            }
            self.rnn_cell = CfcCell(self.in_features, self.hidden_size, hparams)

        # 这里的全连接层用于将输出从 [64, 168, 12] 转换为 [64, 168, 3]
        self.fc = nn.Linear(self.hidden_size, 3)
        # 这里的全连接层用于将输出从 [64, 168, 3] 转换为 [64, 24, 3]
        self.fc_reduce = nn.Linear(self.seq_in_len, self.seq_out_len)  # 用于将时间步从 128 减少到 24

    def forward(self, x, timespans=None, mask=None):
        x = x.squeeze(1)

        x = x.permute(0, 2, 1)
        seq_mean = torch.mean(x, dim=1).unsqueeze(2).permute(0, 2, 1)
        x = x - seq_mean

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

        if self.return_sequences:
            readout = torch.stack(output_sequence, dim=1)
            readout = self.fc(readout)
            readout = readout.permute(0, 2, 1)
            readout = self.fc_reduce(readout)
            readout = readout.permute(0, 2, 1)
            readout = readout + seq_mean_out
        elif mask is not None:
            readout = forwarded_output
        else:
            readout = self.fc(h_state)
        return readout


class LTCCell(nn.Module):
    def __init__(
            self,
            in_features,
            units,
            ode_unfolds=6,
            epsilon=1e-8,
    ):
        super(LTCCell, self).__init__()
        self.in_features = in_features
        self.units = units
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        # self.softplus = nn.Softplus()
        self.softplus = nn.Identity()
        self._allocate_parameters()

    @property
    def state_size(self):
        return self.units

    @property
    def sensory_size(self):
        return self.in_features

    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _erev_initializer(self, shape=None):
        return np.random.default_rng().choice([-1, 1], size=shape)

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.state_size, self.state_size))
            ),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.sensory_size, self.state_size))
            ),
        )

        self._params["input_w"] = self.add_weight(
            name="input_w",
            init_value=torch.ones((self.sensory_size,)),
        )
        self._params["input_b"] = self.add_weight(
            name="input_b",
            init_value=torch.zeros((self.sensory_size,)),
        )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.softplus(self._params["sensory_w"]) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # cm/t is loop invariant
        cm_t = self.softplus(self._params["cm"]).view(1, -1) / (
                (elapsed_time + 1) / self._ode_unfolds
        )

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            w_activation = self.softplus(self._params["w"]) * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = (
                    cm_t * v_pre
                    + self.softplus(self._params["gleak"]) * self._params["vleak"]
                    + w_numerator
            )
            denominator = cm_t + self.softplus(self._params["gleak"]) + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)
            if torch.any(torch.isnan(v_pre)):
                breakpoint()
        return v_pre

    def _map_inputs(self, inputs):
        inputs = inputs * self._params["input_w"]
        inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        output = output * self._params["output_w"]
        output = output + self._params["output_b"]
        return output

    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
        self._params["w"].data = self._clip(self._params["w"].data)
        self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
        self._params["cm"].data = self._clip(self._params["cm"].data)
        self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def forward(self, input, hx, ts):
        # Regularly sampled mode (elapsed time = 1 second)

        batch_size = input.size(0)
        if ts.size(0) != batch_size:
            # Adjust ts if necessary, e.g., repeat last value
            ts = ts[:batch_size]

        ts = ts.view((batch_size, 1))

        inputs = self._map_inputs(input)

        next_state = self._ode_solver(inputs, hx, ts)

        # outputs = self._map_outputs(next_state)

        return next_state
