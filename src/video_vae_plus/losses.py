import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class LPIPS(nn.Module):
    """感知损失：使用预训练VGG16"""
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = vgg16(pretrained=pretrained).features
        self.layers = nn.ModuleList([
            vgg[:4],   # conv1_2
            vgg[4:9],  # conv2_2
            vgg[9:16], # conv3_3
            vgg[16:23] # conv4_3
        ])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        B, C, T, H, W = x.shape
        x_flat = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        y_flat = y.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        x_feats, y_feats = [], []
        for layer in self.layers:
            x_flat = layer(x_flat)
            y_flat = layer(y_flat)
            x_feats.append(x_flat)
            y_feats.append(y_flat)
        losses = [F.mse_loss(xf, yf) for xf, yf in zip(x_feats, y_feats)]
        return sum(losses)


class TemporalConsistencyLoss(nn.Module):
    """时序一致性损失"""
    def forward(self, recon, target):
        recon_grad = recon[:, :, 1:] - recon[:, :, :-1]
        target_grad = target[:, :, 1:] - target[:, :, :-1]
        return F.l1_loss(recon_grad, target_grad)


class MotionGuidedLoss(nn.Module):
    """运动引导损失：在大运动区域加权"""
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def compute_motion_mask(self, video):
        diff = (video[:, :, 1:] - video[:, :, :-1]) ** 2
        flow_mag = torch.sqrt(diff.sum(dim=1) + 1e-8)
        mask = (flow_mag > self.threshold).float()
        return mask

    def forward(self, recon, target):
        mask = self.compute_motion_mask(target)
        # 扩展掩码到时间维度匹配
        mask = mask.unsqueeze(1)  # (B,1,T-1,H,W)
        recon_diff = (recon[:, :, 1:] - recon[:, :, :-1]).abs().mean(dim=1)
        target_diff = (target[:, :, 1:] - target[:, :, :-1]).abs().mean(dim=1)
        weighted_loss = (mask * (recon_diff - target_diff).abs()).mean()
        return weighted_loss


class MultiScaleKLDivergence(nn.Module):
    """多尺度KL散度"""
    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

    def forward(self, mean, logvar):
        total_kl = 0.0
        for scale in range(self.num_scales):
            if scale > 0:
                factor = 2 ** scale
                m = F.avg_pool3d(mean, kernel_size=factor, stride=factor)
                lv = F.avg_pool3d(logvar, kernel_size=factor, stride=factor)
            else:
                m, lv = mean, logvar
            kl = -0.5 * torch.sum(1 + lv - m.pow(2) - lv.exp())
            total_kl += kl / (self.num_scales * m.numel())
        return total_kl


class VideoVAELoss(nn.Module):
    """VideoVAE+ 完整损失"""
    def __init__(self, l1_weight=1.0, lpips_weight=0.1, temporal_weight=0.05,
                 motion_weight=0.1, kl_weight=1e-4):
        super().__init__()
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.temporal_weight = temporal_weight
        self.motion_weight = motion_weight
        self.kl_weight = kl_weight

        self.lpips = LPIPS()
        self.temporal_loss = TemporalConsistencyLoss()
        self.motion_loss = MotionGuidedLoss()
        self.kl_loss = MultiScaleKLDivergence()

    def forward(self, recon, target, mean, logvar, step=None):
        l1 = F.l1_loss(recon, target)
        lpips = self.lpips(recon, target)
        temporal = self.temporal_loss(recon, target)
        motion = self.motion_loss(recon, target)
        kl = self.kl_loss(mean, logvar)

        if step is not None and step < 10000:
            lpips_w = self.lpips_weight * (step / 10000)
        else:
            lpips_w = self.lpips_weight

        total = (self.l1_weight * l1 +
                 lpips_w * lpips +
                 self.temporal_weight * temporal +
                 self.motion_weight * motion +
                 self.kl_weight * kl)
        return total
