# Isaac Sim 完整集成示例

本文档详细介绍如何将 VideoVAE+ 与 CoWVLA 集成到 NVIDIA Isaac Sim 中，实现具身智能机器人的闭环控制。

---

## 1. 环境准备

### 1.1 安装 Isaac Sim

- 推荐使用 **Isaac Sim 2023.1.1** 或更高版本
- 从 NVIDIA Omniverse Launcher 下载并安装
- 安装路径示例：`~/.local/share/ov/pkg/isaac_sim-2023.1.1`

### 1.2 创建 Python 虚拟环境（用于外部脚本）

```bash
conda create --name isaac_vae python=3.10
conda activate isaac_vae
pip install torch torchvision numpy opencv-python

1.3 将 VideoVAE+ 代码添加到 Isaac Sim 的 Python 路径
# 在 Isaac Sim 的扩展目录下创建符号链接
cd ~/.local/share/ov/pkg/isaac_sim-2023.1.1/exts
ln -s /path/to/your/VideoVAEPlus-Insight vae_extension

2. Isaac Sim 扩展开发
2.1 扩展入口文件 extension.toml
在 vae_extension/ 目录下创建 config/extension.toml：
[package]
name = "vae_extension"
version = "1.0.0"
title = "VideoVAE+ Integration"
description = "Integrates VideoVAE+ with Isaac Sim for robot perception and control"

[dependencies]
"omni.kit" = {}
"omni.isaac.core" = {}
"omni.isaac.sensor" = {}

2.2 核心扩展代码 vae_extension/__init__.py
import carb
import omni
import omni.usd
import torch
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera

class VideoVAEExtension:
    def __init__(self):
        self.world = None
        self.robot = None
        self.cameras = {}
        self.vae = None
        self.co_wvla = None
        self.history_buffer = []
        self.buffer_size = 16
        self.control_freq = 30  # Hz
        self.last_control_time = 0
        
        self.init_models()
        self.init_world()
        self.start_control_loop()
    
    def init_models(self):
        """加载预训练模型"""
        from video_vae_plus.model import VideoVAEPlus
        from cowvla.model import CoWVLA
        
        self.vae = VideoVAEPlus.from_pretrained("ckpt/vae_16z_text.pth")
        self.co_wvla = CoWVLA.from_pretrained("ckpt/co_wvla.pth")
        self.vae.cuda()
        self.co_wvla.cuda()
        self.vae.eval()
        self.co_wvla.eval()
        carb.log_info("Models loaded successfully")
    
    def init_world(self):
        """初始化 Isaac Sim 世界"""
        self.world = World(stage_units_in_meters=1.0)
        
        # 加载机器人（以 Franka 为例）
        usd_path = "/isaac/Assets/robots/franka/franka.usd"
        add_reference_to_stage(usd_path, "/World/franka")
        self.robot = self.world.scene.add(
            Robot(prim_path="/World/franka", name="franka")
        )
        
        # 添加相机
        self.add_camera("front", position=(0.5, 0, 0.5), orientation=(0, 0, 0))
        self.add_camera("wrist", position=(0.3, 0, 0.2), orientation=(0, 0, 0))
        
        self.world.scene.add_default_ground_plane()
        carb.log_info("World initialized")
    
    def add_camera(self, name, position, orientation):
        camera_path = f"/World/cameras/{name}"
        camera = Camera(
            prim_path=camera_path,
            position=position,
            orientation=orientation,
            resolution=(256, 256)
        )
        self.cameras[name] = camera
    
    def capture_frame(self, camera_name="front"):
        """捕获 RGB 帧并转换为 PyTorch 张量"""
        camera = self.cameras[camera_name]
        rgb = camera.get_rgb()  # (H, W, 3), range 0-255
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return tensor.cuda()
    
    def update_history(self, frame):
        self.history_buffer.append(frame)
        if len(self.history_buffer) > self.buffer_size:
            self.history_buffer.pop(0)
    
    def get_latent(self):
        if len(self.history_buffer) < self.buffer_size:
            return None
        # (1, C, T, H, W)
        video_tensor = torch.stack(self.history_buffer[-self.buffer_size:], dim=2)
        with torch.no_grad():
            mean, logvar = self.vae.encode(video_tensor)
            latent = self.vae.reparameterize(mean, logvar)
        return latent
    
    def execute_action(self, action):
        """执行动作：action 为 7 维关节角度"""
        # 安全检查：限幅
        action = np.clip(action, -np.pi, np.pi)
        # 发布到机器人
        self.robot.apply_action(action)
    
    def control_loop(self):
        """主控制循环（在单独线程中运行）"""
        import time
        while True:
            now = time.time()
            if now - self.last_control_time < 1.0 / self.control_freq:
                time.sleep(0.001)
                continue
            
            # 1. 捕获当前帧
            frame = self.capture_frame("front")
            self.update_history(frame)
            
            # 2. 获取潜表示
            latent = self.get_latent()
            if latent is None:
                continue
            
            # 3. CoWVLA 推理
            instruction = "pick up the red cube"  # 可改为从外部接口获取
            with torch.no_grad():
                structure, motion = self.co_wvla.disentangle_head(latent)
                motion_chain = self.co_wvla.motion_predictor(
                    instruction_embed=self.encode_text(instruction),
                    current_motion=motion,
                    structure=structure
                )
                actions = self.co_wvla.action_head(motion_chain, structure)
            
            # 4. 执行第一个时间步的动作
            self.execute_action(actions[0, 0].cpu().numpy())
            
            self.last_control_time = now
    
    def encode_text(self, text):
        """文本编码（需实现 Flan-T5 或类似编码器）"""
        # 此处简化，实际应用中可使用 HuggingFace Transformers
        # 返回形状 (1, L, D)
        return torch.zeros(1, 1, 768).cuda()

# 扩展入口
def on_startup():
    carb.log_info("Starting VideoVAE+ extension")
    global _extension
    _extension = VideoVAEExtension()

def on_shutdown():
    carb.log_info("Shutting down VideoVAE+ extension")
3. 启动集成
3.1 在 Isaac Sim 中加载扩展
打开 Isaac Sim

菜单栏 → Window → Extensions

点击齿轮图标 → Add Extension Folder → 选择 vae_extension 所在目录

找到 vae_extension 并启用

3.2 命令行启动（无头模式）
cd ~/.local/share/ov/pkg/isaac_sim-2023.1.1
./python.sh -m isaac_sim --ext-folder /path/to/vae_extension --enable vae_extension

4. ROS2 接口（可选）
如果你希望与真实机器人或 ROS2 生态系统交互，可以添加 ROS2 节点：
# ros2_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch

class VideoVAEROS2Node(Node):
    def __init__(self, vae, co_wvla):
        super().__init__('video_vae_node')
        self.vae = vae
        self.co_wvla = co_wvla
        self.bridge = CvBridge()
        self.history = []
        
        self.sub = self.create_subscription(Image, '/camera/rgb', self.image_callback, 10)
        self.pub = self.create_publisher(String, '/robot/command', 10)
        self.timer = self.create_timer(0.033, self.control_callback)  # 30Hz
    
    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        tensor = torch.from_numpy(cv_img).float() / 255.0
        tensor = tensor.permute(2,0,1).unsqueeze(0)
        self.history.append(tensor)
        if len(self.history) > 16:
            self.history.pop(0)
    
    def control_callback(self):
        if len(self.history) < 16:
            return
        video_tensor = torch.stack(self.history, dim=2).cuda()
        with torch.no_grad():
            latent = self.vae.encode(video_tensor)
            # ... CoWVLA 推理 ...
            actions = self.co_wvla(latent)  # 简化示意
        # 发布动作（例如 JSON 字符串）
        self.pub.publish(String(data=str(actions.cpu().numpy().tolist())))

5. 常见问题
5.1 扩展无法加载
检查 extension.toml 语法

确认 Isaac Sim 版本与扩展要求的依赖匹配

5.2 模型推理速度慢
使用 TensorRT 加速模型推理

降低控制频率至 20Hz

使用更小的潜通道版本（如 4z）

5.3 相机图像获取失败
确保相机 prim 路径正确

检查相机分辨率设置是否与代码一致

6. 效果演示
成功集成后，你将在 Isaac Sim 视口中看到机器人根据视觉输入实时动作。
可在终端看到类似输出：
[INFO] [vae_extension]: Models loaded successfully
[INFO] [vae_extension]: World initialized
[INFO] [vae_extension]: Control loop started at 30 Hz
[INFO] [vae_extension]: Action: [-0.12, 0.34, -0.56, 0.78, -0.23, 0.45, -0.67]
...
至此，VideoVAE+ 与 CoWVLA 已成功集成到 Isaac Sim，实现了从感知到控制的完整闭环。
