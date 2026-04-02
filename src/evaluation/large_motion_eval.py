import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

class MotionEvaluationMetrics:
    def compute_optical_flow(self, video):
        """简化光流计算：帧差近似"""
        diff = video[:, :, 1:] - video[:, :, :-1]
        flow = diff.permute(0,2,1,3,4)
        return flow

    def compute_flow_error(self, recon, target):
        flow_recon = self.compute_optical_flow(recon)
        flow_target = self.compute_optical_flow(target)
        epe = torch.norm(flow_recon - flow_target, dim=2).mean()
        return epe

    def compute_motion_consistency(self, recon, target):
        # 简化：用帧差相关系数
        recon_diff = recon[:, :, 1:] - recon[:, :, :-1]
        target_diff = target[:, :, 1:] - target[:, :, :-1]
        cos_sim = F.cosine_similarity(recon_diff.flatten(1), target_diff.flatten(1), dim=1)
        return cos_sim.mean()

    def compute_motion_blur(self, video):
        fft = torch.fft.fftn(video, dim=(-2,-1))
        high_freq = torch.abs(fft[..., :video.shape[-2]//2, :]).sum()
        total = torch.abs(fft).sum()
        return high_freq / total

    def evaluate_all(self, recon, target):
        return {
            'flow_error': self.compute_flow_error(recon, target).item(),
            'motion_consistency': self.compute_motion_consistency(recon, target).item(),
            'motion_blur_recon': self.compute_motion_blur(recon).item(),
            'motion_blur_target': self.compute_motion_blur(target).item(),
        }

def evaluate_large_motion(model, dataloader):
    metrics = MotionEvaluationMetrics()
    results = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            video = batch['video'].cuda()
            recon, _, _ = model(video)
            scores = metrics.evaluate_all(recon, video)
            results.append(scores)
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    print("Large Motion Evaluation Results:")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")
    return avg
