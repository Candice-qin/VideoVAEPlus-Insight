# CoWVLA 动作预测头设计

## 1. 整体架构

CoWVLA 的核心思想：将 VideoVAE+ 的潜空间解耦为 **结构** 和 **运动**，然后在运动潜空间中进行未来预测，最后映射为机器人动作。

```python
class CoWVLA(nn.Module):
    def __init__(self, vae, action_dim=7):
        super().__init__()
        self.vae = vae  # 预训练 VideoVAE+（冻结）
        self.disentangle_head = DisentangleHead(latent_dim=16)
        self.motion_predictor = MotionPredictor(motion_dim=8)
        self.action_head = ActionHead(motion_dim=8, action_dim=action_dim)
    
    def forward(self, video_segment, instruction):
        # 1. 编码视频 -> 潜变量
        latent = self.vae.encode(video_segment)
        
        # 2. 解耦为结构 + 运动
        structure, motion = self.disentangle_head(latent)
        
        # 3. 预测未来运动链
        motion_chain = self.motion_predictor(
            instruction_embed=self.encode_text(instruction),
            current_motion=motion,
            structure=structure
        )
        
        # 4. 生成动作序列
        actions = self.action_head(motion_chain, structure)
        return actions
2. 潜空间解耦头
python
class DisentangleHead(nn.Module):
    """将VideoVAE+潜变量分解为结构和运动"""
    def __init__(self, latent_dim=16, hidden_dim=128):
        super().__init__()
        # 时空池化聚合信息
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        self.decouple_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出 [structure, motion]
        )
    
    def forward(self, latent):
        # latent: (B, C, T, H, W)
        pooled = self.spatial_pool(latent).squeeze(-1).squeeze(-1)  # (B, C, T)
        pooled = self.temporal_pool(pooled.transpose(1,2)).squeeze(-1)  # (B, C)
        structure, motion = self.decouple_net(pooled).chunk(2, dim=1)
        return structure, motion
3. 运动预测器
python
class MotionPredictor(nn.Module):
    """Transformer-based 运动链预测"""
    def __init__(self, motion_dim=8, hidden_dim=256, num_layers=6):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.input_proj = nn.Linear(motion_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, motion_dim)
    
    def forward(self, instruction_embed, current_motion, structure, future_steps=16):
        # 将指令、当前运动、结构融合为输入序列
        motion_seq = self.input_proj(current_motion.unsqueeze(1))
        # ... transformer 前向
        future = self.output_proj(predicted_seq)
        return future
4. 动作预测头
python
class ActionHead(nn.Module):
    """将运动潜变量映射到机器人控制指令"""
    def __init__(self, motion_dim=8, action_dim=7, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.action_limits = {'min': -np.pi, 'max': np.pi}  # 关节限位
    
    def forward(self, motion_chain, structure):
        # motion_chain: (B, T, motion_dim)
        B, T, _ = motion_chain.shape
        structure_expanded = structure.unsqueeze(1).expand(-1, T, -1)
        combined = torch.cat([motion_chain, structure_expanded], dim=-1)
        actions = self.net(combined)
        # 映射到实际动作范围
        actions = torch.tanh(actions)  # [-1,1]
        actions = self.action_limits['min'] + (actions + 1)/2 * (self.action_limits['max'] - self.action_limits['min'])
        return actions
5. 关键设计优势
高效推理：在低维潜空间（~10²维）做预测，而非像素空间（10⁶维）

物理先验：VideoVAE+ 的潜空间已内嵌运动规律，预测更符合物理

结构-运动解耦：显式建模静态与动态，增强因果推理能力
