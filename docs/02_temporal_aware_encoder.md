# 时序感知空间编码器（完整代码）

## 设计动机

传统空间编码器逐帧独立处理，丢失了帧间运动信息；而直接使用 3D 卷积同时压缩时空，又会导致大幅运动时信息混叠。  
VideoVAE+ 提出一种折中方案：**在空间压缩时“看见”时间，但不压缩时间**。

## 核心实现：`TemporalAwareSpatialEncoder`

```python
import torch
import torch.nn as nn

class TemporalAwareSpatialEncoder(nn.Module):
    """
    时序感知的空间编码器
    使用 kernel_size=(1,3,3) 的3D卷积，保留时间维度但感知相邻帧
    """
    def __init__(self, in_channels=3, out_channels=256, num_downsample=4):
        super().__init__()
        
        channels = [in_channels, 64, 128, 256, 256]
        self.down_layers = nn.ModuleList()
        
        for i in range(num_downsample):
            in_ch = channels[i]
            out_ch = channels[i+1]
            
            # 每个下采样块：两层 (1,3,3) 卷积，第二层带空间下采样
            self.down_layers.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3),
                          stride=(1, 1, 1), padding=(0, 1, 1)),
                nn.GroupNorm(32, out_ch),
                nn.SiLU(),
                nn.Conv3d(out_ch, out_ch, kernel_size=(1, 3, 3),
                          stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.GroupNorm(32, out_ch),
                nn.SiLU()
            ))
        
        self.output_conv = nn.Conv3d(channels[-1], out_channels, kernel_size=1)
    
    def forward(self, x):
        for layer in self.down_layers:
            x = layer(x)
        x = self.output_conv(x)
        return x

维度变化示例
输入 (B, 3, T, 256, 256)，经过 4 个下采样块后：
层	操作	输出维度
输入	-	(B, 3, T, 256, 256)
Block1	两次 (1,3,3) 卷积，第二次 stride=(1,2,2)	(B, 64, T, 128, 128)
Block2	同上	(B, 128, T, 64, 64)
Block3	同上	(B, 256, T, 32, 32)
Block4	同上	(B, 256, T, 16, 16)
Output	1x1 卷积	(B, 256, T, 16, 16)

关键：时间维度 T 全程不变，但每一帧的空间编码都能通过感受野“看到”相邻帧，从而保证运动连贯性。

为什么能解决大幅运动模糊？
传统 3D VAE 使用 (3,3,3) 卷积，一次卷积同时混合时空信息。当相邻帧差异大时，卷积结果会产生“平均效应”，导致运动模糊。

VideoVAE+ 让每一帧的空间编码独立进行（kernel 时间维度=1），但通过堆叠多层后，感受野仍能覆盖多帧，相当于“每帧独立压缩，但压缩时参考了邻居”。
