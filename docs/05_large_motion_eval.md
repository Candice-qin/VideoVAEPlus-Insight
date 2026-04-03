大幅运动场景重建质量评估
1. 评估数据集构建
从通用视频数据集中筛选运动强度高的片段：
```python
 class LargeMotionDataset:
     def filter_videos(self, video_dir, min_motion_score=0.3):
         high_motion = []
         for video in os.listdir(video_dir):
             score = self.compute_motion_score(video)
             if score > min_motion_score:
                 high_motion.append((video, score))
         return sorted(high_motion, key=lambda x: x[1], reverse=True)
def compute_motion_score(self, video):
    cap = cv2.VideoCapture(video)
    prev = None
    flows = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))
        prev = gray
    cap.release()
    return np.mean(flows) if flows else 0

2. 专用评估指标
指标    意义  计算方法
 光流误差 (EPE)    运动轨迹重建精度    重建与目标视频光流的端点误差
 运动一致性    运动模式相似度 特征点轨迹的余弦相似度
 动态细节LPIPS    运动区域感知质量    只在运动掩码上计算LPIPS
 运动模糊分数    高频细节保留  频域高频能量占比
class MotionEvaluationMetrics:
     def compute_flow_error(self, recon, target):
         """光流端点误差"""
         flow_recon = self.optical_flow(recon)
         flow_target = self.optical_flow(target)
         return torch.norm(flow_recon - flow_target, dim=2).mean()
def compute_motion_consistency(self, recon, target):
    """运动轨迹相似度"""
    traj_recon = self.extract_trajectory(recon)
    traj_target = self.extract_trajectory(target)
    return F.cosine_similarity(traj_recon, traj_target, dim=-1).mean()

def compute_dynamic_detail(self, recon, target):
    """运动区域感知损失"""
    motion_mask = self.get_motion_mask(target)
    recon_masked = recon * motion_mask
    target_masked = target * motion_mask
    return self.lpips(recon_masked, target_masked)

def compute_motion_blur(self, video):
    """频域模糊分数（越低越清晰）"""
    fft = torch.fft.fftn(video, dim=(-2,-1))
    high_freq = torch.abs(fft[..., :video.shape[-2]//2, :]).sum()
    total = torch.abs(fft).sum()
    return high_freq / total

3. 综合评估流程
def evaluate_large_motion(model, dataset):
     metrics = MotionEvaluationMetrics()
     results = []
for sample in dataset:
    video = sample['video'].cuda()
    recon = model.reconstruct(video)
    scores = {
        'flow_error': metrics.compute_flow_error(recon, video).item(),
        'motion_consistency': metrics.compute_motion_consistency(recon, video).item(),
        'dynamic_detail': metrics.compute_dynamic_detail(recon, video).item(),
        'motion_blur': metrics.compute_motion_blur(recon).item()
    }
    # 综合分数（越低越好）
    scores['overall'] = (scores['flow_error']*0.3 + 
                         (1-scores['motion_consistency'])*0.2 +
                         scores['dynamic_detail']*0.3 +
                         scores['motion_blur']*0.2)
    results.append(scores)

return pd.DataFrame(results).mean()

4. 评估结果示例（预期）
方法    光流误差↓   运动一致性↑  动态细节↓   运动模糊↓   综合↓
 Cosmos Tokenizer    2.34    0.78    0.052   0.31    0.89
 Hunyuan VAE    1.98    0.82    0.048   0.28    0.76
 VideoVAE+    1.52    0.89    0.032   0.22    0.58
