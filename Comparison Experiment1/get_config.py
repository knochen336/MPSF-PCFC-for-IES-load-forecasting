import argparse


class BaseConfig:
    """基类，包含所有模型共享的超参数"""

    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'test'
        self.seq_len = 168
        self.label_len = 12
        self.pred_len = 96
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.dropout = 0.1
        self.use_norm = 1
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.moving_avg = 24
        self.factor = 1


class PatchTSTConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 128
        self.enc_in = 12

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 40
        self.weightdecay = 0.5
        self.decaypatience = 4


class TimeMixerConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.task_name = "long_term_forecast"  # 任务类型，例如：'long_term_forecast'、'short_term_forecast'、'imputation' 等
        self.seq_len = 168  # 输入序列长度 T
        # 目标序列长度 L
        self.pred_len = 24  # 预测时间步长
        self.down_sampling_window = 2  # 下采样窗口大小
        self.channel_independence = False  # 是否进行通道独立处理
        self.e_layers = 3  # 编码器层数
        self.down_sampling_layers = 2  # 下采样层数
        self.enc_in = 12  # 输入通道数
        self.embed = 64  # 嵌入层维度
        self.d_model = 64  # 模型内部表示的维度
        self.freq = 'h'  # 数据的频率（如：'h'表示小时数据）
        self.dropout = 0.1  # Dropout比率
        self.use_norm = 1  # 是否使用归一化（1表示使用）
        self.c_out = 3  # 输出通道数（通常用于预测任务）
        self.num_class = 10  # 分类任务的类别数
        self.moving_avg = 3  # 时间序列分解中的移动平均窗口大小
        self.down_sampling_method = 'avg'  # 下采样方法（'max'、'avg'、'conv'）
        self.d_ff = 64
        self.lr = 0.00004
        self.batchsize = 256
        self.epochs = 35
        self.weightdecay = 0.8
        self.decaypatience = 6


class TimesNetConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Task-related configurations
        self.task_name = "long_term_forecast"  # 任务类型，例如：'long_term_forecast', 'short_term_forecast', 'imputation', 'anomaly_detection', 'classification'
        self.seq_len = 168  # 输入序列长度 T (例如，168表示一周的小时数据)
        self.label_len = 96  # 目标序列长度 L (用于预测的时间步数)
        self.pred_len = 96  # 预测时间步长
        self.e_layers = 2  # 编码器层数
        self.d_model = 32  # 模型的维度（例如：每个时间步的特征数量）
        self.dropout = 0.1  # Dropout比率
        self.freq = 'h'  # 数据频率（例如：'h'表示小时数据）

        # Embedding-related configurations
        self.enc_in = 12  # 输入特征的数量
        self.embed = 16  # 嵌入层的维度
        self.use_norm = 1  # 是否使用归一化（1表示使用）

        # Output-related configurations
        self.c_out = 3  # 输出通道数（通常用于回归任务）
        # self.num_class = 10  # 分类任务的类别数（如果是分类任务使用）
        self.num_kernels = 4
        # Time series specific configurations
        self.moving_avg = 3  # 时间序列分解中的移动平均窗口大小
        self.down_sampling_window = 2  # 下采样窗口大小
        self.channel_independence = False  # 是否进行通道独立处理
        self.down_sampling_layers = 2  # 下采样层数
        self.down_sampling_method = 'avg'  # 下采样方法（'max', 'avg', 'conv'）
        self.top_k = 5

        # Feed-forward layer dimension
        self.d_ff = 16  # Feed-forward层的维度

        self.lr = 0.00008
        self.batchsize = 256
        self.epochs = 40
        self.weightdecay = 0.8
        self.decaypatience = 5


class TSMixerConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Task-related configurations
        self.task_name = "long_term_forecast"  # 任务类型，例如：'long_term_forecast', 'short_term_forecast', 'imputation', 'anomaly_detection', 'classification'
        self.seq_len = 168  # 输入序列长度 T (例如，168表示一周的小时数据)
        self.label_len = 96  # 目标序列长度 L (用于预测的时间步数)
        self.pred_len = 96  # 预测时间步长
        self.e_layers = 2  # 编码器层数
        self.d_model = 64  # 模型的维度（例如：每个时间步的特征数量）
        self.dropout = 0.2  # Dropout比率
        self.freq = 'h'  # 数据频率（例如：'h'表示小时数据）

        # 输入和输出的通道数，通常在时间序列预测任务中，这会是特征的数量
        self.enc_in = 12  # 输入特征维度 (例如：7个通道或7种特征)
        self.dec_in = 12  # 如果有解码器输入，通常与编码器相同

        self.lr = 0.0008
        self.batchsize = 256
        self.epochs = 40
        self.weightdecay = 0.5
        self.decaypatience = 4


class FITSConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.enc_in = 12
        self.seq_len = 168
        self.pred_len = 96
        self.individual = True

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 50
        self.weightdecay = 0.5
        self.decaypatience = 5


class CFCConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.in_features = 12
        self.hidden_size = 64
        self.out_feature = 3
        self.seq_in_len = 168
        self.seq_out_len =96
        self.individual = True
        self.batch = 256
        self.pred_len = self.seq_out_len

        self.lr = 0.001
        self.batchsize = 256
        self.epochs = 40
        self.weightdecay = 0.8
        self.decaypatience = 5


def get_config(model_name: str):
    """根据模型名称获取特定的超参数配置"""
    if model_name == 'PatchTST':
        return PatchTSTConfig()
    if model_name == 'TimeMixer':
        return TimeMixerConfig()
    if model_name == 'TimesNet':
        return TimesNetConfig()
    if model_name == 'TSMixer':
        return TSMixerConfig()
    if model_name == 'FITS'or 'FITSflops':
        return FITSConfig()
    if model_name == 'CFC':
        return CFCConfig()


    else:
        raise ValueError(f"Unsupported model name: {model_name}")
