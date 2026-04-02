import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TemporalEncoder(nn.Module):
    """轻量级时序编码器，压缩时间维度"""
    def __init__(self, in_channels=256, latent_dim=16):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 128, kernel_size=(3, 1, 1),
                               stride=(2, 1, 1), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(128, 64, kernel_size=(3, 1, 1),
                               stride=(2, 1, 1), padding=(1, 0, 0))
        self.to_latent = nn.Conv3d(64, latent_dim * 2, kernel_size=1)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        mean, logvar = self.to_latent(x).chunk(2, dim=1)
        return mean, logvar


class TemporalDecoder(nn.Module):
    """时序解码器"""
    def __init__(self, latent_dim=16, out_channels=256):
        super().__init__()
        self.from_latent = nn.Conv3d(latent_dim, 64, kernel_size=1)
        self.conv1 = nn.ConvTranspose3d(64, 128, kernel_size=(3, 1, 1),
                                        stride=(2, 1, 1), padding=(1, 0, 0), output_padding=(1, 0, 0))
        self.conv2 = nn.ConvTranspose3d(128, out_channels, kernel_size=(3, 1, 1),
                                        stride=(2, 1, 1), padding=(1, 0, 0), output_padding=(1, 0, 0))

    def forward(self, z):
        x = F.silu(self.from_latent(z))
        x = F.silu(self.conv1(x))
        x = self.conv2(x)
        return x


class SpatialDecoder(nn.Module):
    """空间解码器，与编码器对称"""
    def __init__(self, in_channels=256, out_channels=3, num_upsample=4):
        super().__init__()
        channels = [in_channels, 256, 128, 64, out_channels]
        self.up_layers = nn.ModuleList()
        for i in range(num_upsample):
            in_ch = channels[i]
            out_ch = channels[i+1]
            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 3, 3),
                                   stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1)),
                nn.GroupNorm(32, out_ch) if out_ch > 3 else nn.Identity(),
                nn.SiLU() if out_ch > 3 else nn.Identity()
            ))

    def forward(self, x):
        for layer in self.up_layers:
            x = layer(x)
        return x


class VideoVAEPlus(nn.Module):
    """完整的 VideoVAE+ 模型"""
    def __init__(self, latent_channels=16, use_text_guidance=True):
        super().__init__()
        self.latent_channels = latent_channels
        self.use_text_guidance = use_text_guidance

        self.spatial_encoder = TemporalAwareSpatialEncoder(
            in_channels=3, out_channels=256, num_downsample=4)
        self.temporal_encoder = TemporalEncoder(
            in_channels=256, latent_dim=latent_channels)
        self.temporal_decoder = TemporalDecoder(
            latent_dim=latent_channels, out_channels=256)
        self.spatial_decoder = SpatialDecoder(
            in_channels=256, out_channels=3, num_upsample=4)

        # 文本引导的交叉注意力（简化版，实际需要 Flan-T5 编码器）
        if use_text_guidance:
            self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

    def encode(self, x):
        """编码视频到潜变量"""
        spatial_feat = self.spatial_encoder(x)       # (B,256,T,H/8,W/8)
        mean, logvar = self.temporal_encoder(spatial_feat)  # (B,latent,T/4,H/8,W/8)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, text_embeds=None):
        """从潜变量解码视频"""
        temporal_feat = self.temporal_decoder(z)     # (B,256,T,H/8,W/8)
        if text_embeds is not None and self.use_text_guidance:
            # 简化：将文本特征融合（实际应插入解码器各层）
            B, C, T, H, W = temporal_feat.shape
            feat_flat = temporal_feat.flatten(2).permute(0,2,1)  # (B, T*H*W, C)
            attn_out, _ = self.cross_attn(feat_flat, text_embeds, text_embeds)
            temporal_feat = attn_out.permute(0,2,1).reshape(B, C, T, H, W)
        recon = self.spatial_decoder(temporal_feat)
        return recon

    def forward(self, x, text_embeds=None):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z, text_embeds)
        return recon, mean, logvar
