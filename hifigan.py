import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from utils import *
from norm2d import *


# 定义 LeakyReLU 的参数
LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    """
    ResBlock1 类实现了一个残差块（Residual Block），该残差块包含多个膨胀卷积层和跳跃连接。
    该模块通过堆叠多个卷积层和激活函数，逐步增加感受野，同时通过残差连接保持信息的流动。

    参数说明:
        cfg: 配置参数对象。
        channels (int): 输入和输出的通道数。
        kernel_size (int, 可选): 卷积核大小，默认为3。
        dilation (Tuple[int, int, int], 可选): 膨胀因子列表，默认为 (1, 3, 5)。
    """
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        # 配置参数
        self.cfg = cfg

        # 定义第一组卷积层列表
        self.convs1 = nn.ModuleList(
            [
                weight_norm( # 应用权重归一化
                    Conv1d(  # 创建 1D 卷积层
                        channels,  # 输入通道数
                        channels,  # 输出通道数
                        kernel_size,  # 卷积核大小
                        1,  # 步长
                        dilation=dilation[0],  # 膨胀因子
                        padding=get_padding(kernel_size, dilation[0]),  # 计算填充大小
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        # 初始化卷积层的权重
        self.convs1.apply(init_weights)

        # 定义第二组卷积层列表
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        # 初始化卷积层的权重
        self.convs2.apply(init_weights)

    def forward(self, x):
        """
        前向传播方法，执行残差块的前向计算。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 输出张量。
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            # 应用 LeakyReLU 激活函数
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 通过第一组卷积层
            xt = c1(xt)
            # 再次应用 LeakyReLU 激活函数
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            # 通过第二组卷积层
            xt = c2(xt)
            # 残差连接
            x = xt + x
        return x

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        for l in self.convs1:
            # 移除第一组卷积层的权重归一化
            remove_weight_norm(l)
        for l in self.convs2:
            # 移除第二组卷积层的权重归一化
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    """
    ResBlock2 类实现了一个残差块（Residual Block），该残差块包含两个膨胀卷积层和跳跃连接。
    该模块通过堆叠两个卷积层和激活函数，逐步增加感受野，同时通过残差连接保持信息的流动。

    参数说明:
        cfg: 配置参数对象。
        channels (int): 输入和输出的通道数。
        kernel_size (int, 可选): 卷积核大小，默认为3。
        dilation (Tuple[int, int], 可选): 膨胀因子列表，默认为 (1, 3)。
    """
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        # 配置参数
        self.cfg = cfg

        # 定义卷积层列表
        self.convs = nn.ModuleList(
            [
                weight_norm(  # 应用权重归一化
                    Conv1d(  # 创建 1D 卷积层
                        channels,  # 输入通道数
                        channels,  # 输出通道数
                        kernel_size,  # 卷积核大小
                        1,  # 步长
                        dilation=dilation[0],  # 膨胀因子
                        padding=get_padding(kernel_size, dilation[0]),  # 计算填充大小
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        # 初始化卷积层的权重
        self.convs.apply(init_weights)

    def forward(self, x):
        """
        前向传播方法，执行残差块的前向计算。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 输出张量。
        """
        for c in self.convs:
            # 应用 LeakyReLU 激活函数
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 通过卷积层
            xt = c(xt)
            # 残差连接
            x = xt + x
        return x

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        for l in self.convs:
            # 移除卷积层的权重归一化
            remove_weight_norm(l)


class HiFiGAN(torch.nn.Module):
    """
    HiFiGAN 类实现了一个基于生成对抗网络（GAN）的音频生成模型。
    该模型通过多个上采样层和残差块，逐步将低分辨率的梅尔频谱转换为高分辨率的音频信号。
    HiFiGAN 结合了高保真生成能力和高效的计算资源利用，广泛应用于语音合成和音频生成任务。

    参数说明:
        cfg: 配置参数对象，包含以下字段:
            preprocess:
                n_mel (int): 梅尔频谱的维度数。
            model:
                hifigan:
                    resblock (str): 残差块类型，'1' 表示使用 ResBlock1，'2' 表示使用 ResBlock2。
                    upsample_rates (List[int]): 上采样率列表。
                    upsample_kernel_sizes (List[int]): 上采样卷积核大小列表。
                    upsample_initial_channel (int): 初始上采样通道数。
                    resblock_kernel_sizes (List[int]): 残差块卷积核大小列表。
                    resblock_dilation_sizes (List[Tuple[int, ...]]): 残差块膨胀因子列表。
    """
    def __init__(self, cfg):
        super(HiFiGAN, self).__init__()
        # 配置参数
        self.cfg = cfg

        # 残差块卷积核数量
        self.num_kernels = len(self.cfg.model.hifigan.resblock_kernel_sizes)
        # 上采样层数量
        self.num_upsamples = len(self.cfg.model.hifigan.upsample_rates)

        # 初始卷积层，使用权重归一化
        self.conv_pre = weight_norm(
            Conv1d(
                cfg.preprocess.n_mel, # 输入通道数（梅尔频谱维度）
                self.cfg.model.hifigan.upsample_initial_channel, # 输出通道数（初始上采样通道数）
                7, # 卷积核大小
                1, # 步长
                padding=3, # 填充大小
            )
        )
        # 根据配置选择残差块类型
        resblock = ResBlock1 if self.cfg.model.hifigan.resblock == "1" else ResBlock2

        # 上采样层列表
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(
                self.cfg.model.hifigan.upsample_rates, # 上采样率列表
                self.cfg.model.hifigan.upsample_kernel_sizes, # 上采样卷积核大小列表
            )
        ):
            # 创建并添加上采样卷积层，使用权重归一化
            self.ups.append(
                weight_norm( 
                    ConvTranspose1d(
                        self.cfg.model.hifigan.upsample_initial_channel // (2**i),
                        self.cfg.model.hifigan.upsample_initial_channel
                        // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        # 残差块层列表
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            # 当前通道数
            ch = self.cfg.model.hifigan.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(
                    self.cfg.model.hifigan.resblock_kernel_sizes, # 残差块卷积核大小列表
                    self.cfg.model.hifigan.resblock_dilation_sizes, # 残差块膨胀因子列表
                )
            ):
                # 创建并添加残差块
                self.resblocks.append(resblock(self.cfg, ch, k, d))

        # 最后的卷积层，使用权重归一化
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        # 应用权重初始化
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """
        前向传播方法，执行 HiFi-GAN 的前向计算。

        参数:
            x (Tensor): 输入梅尔频谱，形状为 (B, n_mel, T)。

        返回:
            Tensor: 生成的高分辨率音频信号，形状为 (B, 1, T * prod(upsample_rates))。
        """
        # 通过初始卷积层
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            # 应用 LeakyReLU 激活函数
            x = F.leaky_relu(x, LRELU_SLOPE)
            # 通过上采样层
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    # 通过残差块
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    # 累积残差块输出
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # 平均残差块输出
            x = xs / self.num_kernels
        # 应用 LeakyReLU 激活函数
        x = F.leaky_relu(x)
        # 通过最后的卷积层
        x = self.conv_post(x)
        # 应用 tanh 激活函数
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        print("Removing weight norm...")
        for l in self.ups:
            # 移除上采样层的权重归一化
            remove_weight_norm(l)
        for l in self.resblocks:
            # 移除残差块的权重归一化
            l.remove_weight_norm()
        # 移除初始卷积层的权重归一化
        remove_weight_norm(self.conv_pre)
        # 移除最后卷积层的权重归一化
        remove_weight_norm(self.conv_post)


class ResBlock1_vits(torch.nn.Module):
    """
    ResBlock1_vits 类实现了一个残差块（Residual Block），该残差块包含多个膨胀卷积层和跳跃连接。
    该模块通过堆叠多个卷积层和激活函数，逐步增加感受野，同时通过残差连接保持信息的流动。
    该类支持在卷积操作中应用掩码，以实现因果卷积或掩码卷积。

    参数说明:
        channels (int): 输入和输出的通道数。
        kernel_size (int, 可选): 卷积核大小，默认为3。
        dilation (Tuple[int, int, int], 可选): 膨胀因子列表，默认为 (1, 3, 5)。
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1_vits, self).__init__()
        # 定义第一组卷积层列表
        self.convs1 = nn.ModuleList(
            [
                weight_norm(  # 应用权重归一化
                    Conv1d(  # 创建 1D 卷积层
                        channels,  # 输入通道数
                        channels,  # 输出通道数
                        kernel_size,  # 卷积核大小
                        1,  # 步长
                        dilation=dilation[0],  # 膨胀因子
                        padding=get_padding(kernel_size, dilation[0]),  # 计算填充大小
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        # 初始化卷积层的权重
        self.convs1.apply(init_weights)

        # 定义第二组卷积层列表
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        # 初始化卷积层的权重
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        """
        前向传播方法，执行残差块的前向计算。

        参数:
            x (Tensor): 输入张量。
            x_mask (Tensor, 可选): 输入掩码张量，用于掩码卷积操作。

        返回:
            Tensor: 输出张量。
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            # 应用 LeakyReLU 激活函数
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                # 应用掩码
                xt = xt * x_mask
            # 通过第一组卷积层
            xt = c1(xt)

            # 再次应用 LeakyReLU 激活函数
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                # 应用掩码
                xt = xt * x_mask
            # 通过第二组卷积层
            xt = c2(xt)

            # 残差连接
            x = xt + x

        if x_mask is not None:
            # 应用掩码
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        for l in self.convs1:
            # 移除第一组卷积层的权重归一化
            remove_weight_norm(l)
        for l in self.convs2:
            # 移除第二组卷积层的权重归一化
            remove_weight_norm(l)


class ResBlock2_vits(torch.nn.Module):
    """
    ResBlock2_vits 类实现了一个残差块（Residual Block），该残差块包含两个膨胀卷积层和跳跃连接。
    该模块通过堆叠两个卷积层和激活函数，逐步增加感受野，同时通过残差连接保持信息的流动。
    该类支持在卷积操作中应用掩码，以实现因果卷积或掩码卷积。

    参数说明:
        channels (int): 输入和输出的通道数。
        kernel_size (int, 可选): 卷积核大小，默认为3。
        dilation (Tuple[int, int], 可选): 膨胀因子列表，默认为 (1, 3)。
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2_vits, self).__init__()

        # 定义卷积层列表
        self.convs = nn.ModuleList(
            [
                weight_norm(  # 应用权重归一化
                    Conv1d(  # 创建 1D 卷积层
                        channels,  # 输入通道数
                        channels,  # 输出通道数
                        kernel_size,  # 卷积核大小
                        1,  # 步长
                        dilation=dilation[0],  # 膨胀因子
                        padding=get_padding(kernel_size, dilation[0]),  # 计算填充大小
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        # 初始化卷积层的权重
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        """
        前向传播方法，执行残差块的前向计算。

        参数:
            x (Tensor): 输入张量。
            x_mask (Tensor, 可选): 输入掩码张量，用于掩码卷积操作。

        返回:
            Tensor: 输出张量。
        """
        for c in self.convs:
            # 应用 LeakyReLU 激活函数
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                # 应用掩码
                xt = xt * x_mask
            # 通过卷积层
            xt = c(xt)
            # 残差连接
            x = xt + x
        if x_mask is not None:
            # 应用掩码
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        for l in self.convs:
            # 移除卷积层的权重归一化
            remove_weight_norm(l)


class HiFiGAN_vits(torch.nn.Module):
    """
    HiFiGAN_vits 类实现了一个基于生成对抗网络（GAN）的音频生成模型。
    该模型通过多个上采样层和残差块，逐步将低分辨率的特征图转换为高分辨率的音频信号。
    HiFiGAN_vits 结合了高保真生成能力和高效的计算资源利用，广泛应用于语音合成和音频生成任务。
    该类支持使用全局条件输入（g）来指导生成过程。

    参数说明:
        initial_channel (int): 初始输入通道数。
        resblock (str): 残差块类型，'1' 表示使用 ResBlock1_vits，'2' 表示使用 ResBlock2_vits。
        resblock_kernel_sizes (List[int]): 残差块卷积核大小列表。
        resblock_dilation_sizes (List[Tuple[int, ...]]): 残差块膨胀因子列表。
        upsample_rates (List[int]): 上采样率列表。
        upsample_initial_channel (int): 初始上采样通道数。
        upsample_kernel_sizes (List[int]): 上采样卷积核大小列表。
        gin_channels (int, 可选): 全局条件输入的通道数，默认为0。
    """
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(HiFiGAN_vits, self).__init__()
        # 残差块卷积核数量
        self.num_kernels = len(resblock_kernel_sizes)
        # 上采样层数量
        self.num_upsamples = len(upsample_rates)

        # 初始卷积层
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        # 根据配置选择残差块类型
        resblock = ResBlock1_vits if resblock == "1" else ResBlock2_vits

        # 上采样层列表
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # 创建并添加上采样卷积层，使用权重归一化
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # 残差块层列表
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            # 当前通道数
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                # 创建并添加残差块
                self.resblocks.append(resblock(ch, k, d))

        # 最后的卷积层
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # 应用权重初始化
        self.ups.apply(init_weights)

        if gin_channels != 0:
            # 如果存在全局条件输入，则添加一个1D卷积层用于处理条件输入
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        """
        前向传播方法，执行 HiFiGAN_vits 的前向计算。

        参数:
            x (Tensor): 输入特征图，形状为 (B, initial_channel, T)。
            g (Tensor, 可选): 全局条件输入，形状为 (B, gin_channels, T)。

        返回:
            Tensor: 生成的高分辨率音频信号，形状为 (B, 1, T * prod(upsample_rates))。
        """
        # 通过初始卷积层
        x = self.conv_pre(x)
        if g is not None:
            # 如果存在全局条件输入，则添加条件信息
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            # 应用 LeakyReLU 激活函数
            x = F.leaky_relu(x, LRELU_SLOPE)
            # 通过上采样层
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    # 通过残差块
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    # 累积残差块输出
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # 平均残差块输出
            x = xs / self.num_kernels

        # 应用 LeakyReLU 激活函数
        x = F.leaky_relu(x)
        # 通过最后的卷积层
        x = self.conv_post(x)
        # 应用 tanh 激活函数
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        for l in self.ups:
            # 移除上采样层的权重归一化
            remove_weight_norm(l)
        for l in self.resblocks:
            # 移除残差块的权重归一化
            l.remove_weight_norm()
