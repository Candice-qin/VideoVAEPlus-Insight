# 训练数据预处理与损失函数

## 1. 视频-文本对构造流程

### 数据来源
- WebVid-10M
- Panda-70M
- HD-VG-130M

### 预处理步骤

```python
class VideoPreprocessor:
    def process(self, video_path, text):
        # 1. 解码视频
        frames = self.decode_video(video_path)
        
        # 2. 动态场景筛选（保留大幅运动样本）
        motion_score = self.compute_motion_score(frames)
        if motion_score < threshold:
            return None  # 跳过静态视频
        
        # 3. 时间轴采样（随机抽取16帧连续片段）
        indices = self.sample_continuous_frames(len(frames), target_frames=16)
        
        # 4. 缩放到固定分辨率 (256x256)
        resized_frames = [cv2.resize(frames[i], (256,256)) for i in indices]
        
        # 5. 归一化到 [-1, 1]
        tensor = torch.stack([self.normalize(f) for f in resized_frames])
        return tensor.permute(1,0,2,3)  # (C, T, H, W)
    
    def compute_motion_score(self, frames):
        """计算光流幅值作为运动强度"""
        flows = []
        for i in range(len(frames)-1):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
            flows.append(magnitude)
        return np.mean(flows)

2. 完整损失函数实现
class VideoVAELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1_weight = config.l1_weight      # 1.0
        self.lpips_weight = config.lpips_weight  # 0.1
        self.temporal_weight = config.temporal_weight  # 0.05
        self.motion_weight = config.motion_weight  # 0.1
        self.kl_weight = config.kl_weight      # 1e-4
        
        self.lpips = LPIPS()  # 预训练VGG16特征感知损失
        self.temporal_loss = TemporalConsistencyLoss()
        self.motion_loss = MotionGuidedLoss()
        self.kl_loss = MultiScaleKLDivergence()
    
    def forward(self, recon, target, mean, logvar, step=None):
        # 基础重建损失
        l1 = F.l1_loss(recon, target)
        
        # 感知损失
        lpips = self.lpips(recon, target)
        
        # 时序一致性损失（相邻帧差异）
        temporal = self.temporal_loss(recon, target)
        
        # 运动引导损失（在大运动区域加权）
        motion = self.motion_loss(recon, target)
        
        # 多尺度KL散度
        kl = self.kl_loss(mean, logvar)
        
        # 动态权重（早期降低感知损失）
        if step and step < 10000:
            lpips_weight = self.lpips_weight * (step / 10000)
        else:
            lpips_weight = self.lpips_weight
        
        total = (self.l1_weight * l1 +
                 lpips_weight * lpips +
                 self.temporal_weight * temporal +
                 self.motion_weight * motion +
                 self.kl_weight * kl)
        
        return total
关键子损失说明
时序一致性损失：计算重建视频与目标视频的帧间差异的L1距离，确保运动轨迹平滑。

运动引导损失：计算光流作为运动显著性掩码，在大运动区域施加更高权重，让模型更关注动态区域。

多尺度KL散度：在不同空间分辨率下计算KL，使潜空间在不同尺度上都保持平滑。
