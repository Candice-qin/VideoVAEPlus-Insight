import torch
import numpy as np
import cv2

def compute_psnr(recon, target):
    """计算 PSNR，假设输入范围 [-1,1]"""
    mse = torch.mean((recon - target) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

def compute_ssim(recon, target):
    """简化 SSIM，实际建议使用 torchmetrics"""
    # 占位实现，可后续替换
    return torch.tensor(0.0)

def save_video(tensor, path, fps=8):
    """保存视频张量 (C,T,H,W) 范围 [-1,1] 到文件"""
    tensor = (tensor + 1) / 2 * 255  # 转 0-255
    tensor = tensor.permute(1,2,3,0).cpu().numpy().astype(np.uint8)
    h, w = tensor.shape[1], tensor.shape[2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for t in range(tensor.shape[0]):
        out.write(tensor[t][..., ::-1])  # RGB to BGR
    out.release()
