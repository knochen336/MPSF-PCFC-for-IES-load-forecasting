import argparse


class BaseConfig:
    """基类，包含所有模型共享的超参数"""

    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'test'

        self.seq_len = 168
        self.pred_len = 96
        self.label_len = self.pred_len

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
        self.d_layers = 1
        self.d_ff = 128
        self.expand = 2
        self.d_conv = 4
        self.top_k = 5
        self.num_kernels = 6
        self.enc_in = 12
        self.dec_in = 7
        self.c_out = 7

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 40
        self.weightdecay = 0.5
        self.decaypatience = 4


class Dlinear(BaseConfig):
    def __init__(self):
        super().__init__()

        self.enc_in = 12
        self.moving_avg = 25

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 100
        self.weightdecay = 0.5
        self.decaypatience = 4



class TiDE(BaseConfig):
    def __init__(self):
        super().__init__()

        self.enc_in = 12


        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 100
        self.weightdecay = 0.5
        self.decaypatience = 4


class Autoformer(BaseConfig):
    def __init__(self):
        super().__init__()

        self.enc_in = 12  # 输入特征维度（假设只有一个目标变量）
        self.dec_in = 12  # 解码器输入维度
        self.d_model = 128  # 模型的隐藏层维度
        self.batch_size = 256  # 批次大小
        self.embed = 'linear'  # 嵌入类型（线性）
        self.freq = 'h'  # 数据频率
        self.factor = 1  # 自注意力机制的因子
        self.output_attention = False  # 不输出注意力
        self.moving_avg = 25  # 趋势分解的滑动平均窗口
        self.e_layers = 2  # 编码器层数
        self.d_layers = 2  # 解码器层数
        self.n_heads = 4  # 多头注意力的头数
        self.d_ff = 128  # 前馈层的维度
        self.activation = 'gelu'  # 激活函数
        self.c_out = 1  # 输出通道数（预测一个目标变量）
        self.num_class = 10  # 分类任务的类别数（如果是分类任务的话）
        self.task_name = 'long_term_forecast'
        self.auxiliary = 15

        # 训练相关参数
        self.lr = 0.008  # 学习率
        self.batchsize = 256  # 训练的批量大小
        self.epochs = 100  # 训练的轮数
        self.weightdecay = 0.5  # 权重衰减系数
        self.decaypatience = 3  # 用于学习率调整等机制中，控制耐心值


class FITS(BaseConfig):
    def __init__(self):
        super().__init__()
        self.enc_in = 12
        self.individual = False

        self.lr = 0.001  # 学习率
        self.batchsize = 256  # 训练的批量大小
        self.epochs = 100  # 训练的轮数
        self.weightdecay = 0.9  # 权重衰减系数
        self.decaypatience = 5  # 用于学习率调整等机制中，控制耐心值


class FiLM(BaseConfig):
    def __init__(self):
        super().__init__()
        self.output_attention = 1
        self.e_layers = 2
        self.enc_in = 12
        self.ratio = 0.5

        self.lr = 0.002  # 学习率
        self.batchsize = 128  # 训练的批量大小
        self.epochs = 50  # 训练的轮数
        self.weightdecay = 0.5  # 权重衰减系数
        self.decaypatience = 3  # 用于学习率调整等机制中，控制耐心值

class NLinear(BaseConfig):
    def __init__(self):
        super().__init__()


        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 100
        self.weightdecay = 0.5
        self.decaypatience = 4

class TimeXer(BaseConfig):
    def __init__(self):
        super().__init__()
        self.features = 'M'
        self.use_norm = True
        self.patch_len = 24
        self.enc_in = 12
        self.d_model = 128
        self.n_heads = 4
        self.e_layers = 2
        self.d_ff = 128
        self.factor = 5
        self.freq = 'h'  # 数据频率

        self.lr = 0.001
        self.batchsize = 256
        self.epochs = 50
        self.weightdecay = 0.5
        self.decaypatience = 3

class iTransformer(BaseConfig):
    def __init__(self):
        super().__init__()
        self.output_attention = False  # 不输出注意力
        self.features = 'M'
        self.use_norm = True
        self.patch_len = 24
        self.enc_in = 12
        self.d_model = 128
        self.n_heads = 4
        self.e_layers = 2
        self.d_ff = 128
        self.factor = 5
        self.freq = 'h'  # 数据频率

        # 训练相关参数
        self.lr = 0.001  # 学习率
        self.batchsize = 256  # 训练的批量大小
        self.epochs = 100  # 训练的轮数
        self.weightdecay = 0.5  # 权重衰减系数
        self.decaypatience = 5  # 用于学习率调整等机制中，控制耐心值


class MambaSimple(BaseConfig):
    def __init__(self):
        super().__init__()
        self.features = 'M'
        self.use_norm = True
        self.patch_len = 24
        self.enc_in = 12
        self.d_model = 128
        self.n_heads = 4
        self.e_layers = 2
        self.d_ff = 128
        self.factor = 5
        self.freq = 'h'  # 数据频率
        self.expand = 2
        self.d_conv = 4
        self.c_out = 12

        # 训练相关参数
        self.lr = 0.001  # 学习率
        self.batchsize = 32  # 训练的批量大小
        self.epochs = 100  # 训练的轮数
        self.weightdecay = 0.5  # 权重衰减系数
        self.decaypatience = 5  # 用于学习率调整等机制中，控制耐心值


class Crossformer(BaseConfig):
    def __init__(self):
        super().__init__()
        self.features = 'M'
        self.enc_in = 12  # 输入特征维度（假设只有一个目标变量）
        self.dec_in = 12  # 解码器输入维度
        self.d_model = 128  # 模型的隐藏层维度
        self.batch_size = 256  # 批次大小
        self.embed = 'linear'  # 嵌入类型（线性）
        self.freq = 'h'  # 数据频率
        self.factor = 1  # 自注意力机制的因子
        self.output_attention = False  # 不输出注意力
        self.moving_avg = 25  # 趋势分解的滑动平均窗口
        self.e_layers = 2  # 编码器层数
        self.d_layers = 2  # 解码器层数
        self.n_heads = 4  # 多头注意力的头数
        self.d_ff = 128  # 前馈层的维度
        self.activation = 'gelu'  # 激活函数
        self.c_out = 1  # 输出通道数（预测一个目标变量）
        self.num_class = 10  # 分类任务的类别数（如果是分类任务的话）
        self.task_name = 'long_term_forecast'
        self.auxiliary = 15


        # 训练相关参数
        self.lr = 0.001  # 学习率
        self.batchsize = 256  # 训练的批量大小
        self.epochs = 30  # 训练的轮数
        self.weightdecay = 0.5  # 权重衰减系数
        self.decaypatience = 1  # 用于学习率调整等机制中，控制耐心值

class SparseTSF(BaseConfig):
    def __init__(self):
        super().__init__()

        self.enc_in = 12
        self.period_len = 24


        # 训练相关参数
        self.lr = 0.001  # 学习率
        self.batchsize = 256  # 训练的批量大小
        self.epochs = 100  # 训练的轮数
        self.weightdecay = 0.8  # 权重衰减系数
        self.decaypatience = 3  # 用于学习率调整等机制中，控制耐心值


class xlstm(BaseConfig):
    def __init__(self):
        super().__init__()
        self.context_points = 168
        self.target_points = 24
        self.enc_in = 12
        self.n2 = 256
        self.embedding_dim = 256



        self.lr = 0.001  # 学习率
        self.batchsize = 256  # 训练的批量大小
        self.epochs = 100  # 训练的轮数
        self.weightdecay = 0.8  # 权重衰减系数
        self.decaypatience = 3  # 用于学习率调整等机制中，控制耐心值

def get_config(model_name: str):
    """根据模型名称获取特定的超参数配置"""
    if model_name == 'PatchTST':#0
        return PatchTSTConfig()
    elif model_name == 'DLinear':#1
        return Dlinear()
    elif model_name == 'TiDE':#0
        return TiDE()
    elif model_name == 'Autoformer':#0
        return Autoformer()
    elif model_name == 'FITS':#0
        return FITS()
    elif model_name == 'FiLM':#1
        return FiLM()
    elif model_name == 'NLinear':#1
        return NLinear()
    elif model_name == 'TimeXer':#1
        return TimeXer()
    elif model_name == 'iTransformer':#1
        return iTransformer()
    elif model_name == 'MambaSimple':#0
        return MambaSimple()
    elif model_name == 'Crossformer':#1
        return Crossformer()
    elif model_name == 'SparseTSF':#1
        return SparseTSF()
    elif model_name == 'xlstm':
        return xlstm()


    else:
        raise ValueError(f"Unsupported model name: {model_name}")
