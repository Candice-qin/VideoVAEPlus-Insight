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
        self.timer = self.create_timer(0.033, self.control_callback)

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
            # 后续推理...
        # 发布动作
        self.pub.publish(String(data="action_placeholder"))
