# VideoVAE+ Insight: 深度技术解读与具身智能部署实践

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> 本项目是对 **[VideoVAE+](https://github.com/VideoVerses/VideoVAEPlus)** (ICCV 2025) 的深度技术解读、二次开发与具身智能部署实践。  
> 我们不仅复现了原模型的训练与推理，更深入剖析了其核心设计（时空分离编码、文本引导、多尺度KL等），并进一步将潜空间与 **CoWVLA** 结合，在 **Isaac Sim** 中完成了机器人闭环控制的全链路实现。

---

## ✨ 亮点

- 🔍 **深度拆解**：逐模块解析 VideoVAE+ 的架构创新，附完整代码注释
- 🧠 **潜空间应用**：展示如何将 VideoVAE+ 的潜变量解耦为“结构+运动”，用于具身智能世界模型（CoWVLA）
- 🤖 **仿真部署**：提供 Isaac Sim 集成方案，实现真实机器人闭环控制
- 📊 **专用评估**：设计大幅运动场景下的评估指标（光流误差、运动一致性等）
- 🛠️ **工程化代码**：训练、推理、评估、部署全流程脚本，开箱即用

---

## 📖 目录

- [快速开始](#快速开始)
- [技术深度解读](#技术深度解读)
- [具身智能部署](#具身智能部署)
- [项目结构](#项目结构)
- [引用与致谢](#引用与致谢)
- [许可证](#许可证)

---

## 🚀 快速开始

### 环境配置

```bash
git clone https://github.com/yourname/VideoVAEPlus-Insight.git
cd VideoVAEPlus-Insight
conda create --name vae_insight python=3.10 -y
conda activate vae_insight
pip install -r requirements.txt
```

### 下载预训练权重

从 [VideoVAE+ 官方仓库](https://github.com/VideoVerses/VideoVAEPlus) 下载所需权重（如 `sota-16z-text.pth`），放置于 `ckpt/` 目录。

### 运行推理示例

```bash
# 视频重建
bash scripts/run_inference_video.sh

# 图像重建
bash scripts/run_inference_image.sh
```

### 训练模型

```bash
bash scripts/run_training.sh config_16z
```

---

## 📚 技术深度解读

以下文档按模块拆解，每篇均包含**设计动机、核心代码实现、训练/评估细节**：

- [整体架构与设计思想](docs/01_architecture_overview.md)
- [时序感知空间编码器（完整代码）](docs/02_temporal_aware_encoder.md)
- [训练数据预处理与损失函数](docs/03_training_pipeline.md)
- [CoWVLA 动作预测头设计](docs/04_cowvla_action_head.md)
- [大幅运动场景重建质量评估](docs/05_large_motion_eval.md)

---

## 🤖 具身智能部署

我们展示了如何将 VideoVAE+ 的潜变量解耦为**结构潜变量**（静态场景）和**运动潜变量**（动态轨迹），并利用它们构建世界模型预测链。

- [Isaac Sim 完整集成示例](docs/06_isaac_sim_integration.md)
- 支持 ROS2 接口，可直接用于真实机械臂

---

## 📁 项目结构

```
VideoVAEPlus-Insight/
├── README.md                   # 项目主页
├── docs/                       # 深度技术文档
│   ├── 01_architecture_overview.md
│   ├── 02_temporal_aware_encoder.md
│   ├── 03_training_pipeline.md
│   ├── 04_cowvla_action_head.md
│   ├── 05_large_motion_eval.md
│   └── 06_isaac_sim_integration.md
├── src/                        # 可复用代码模块
│   ├── video_vae_plus/         # 模型实现（带注释）
│   ├── training/               # 训练脚本
│   ├── evaluation/             # 评估脚本
│   └── deployment/             # Isaac Sim / ROS2 部署
├── scripts/                    # 一键运行脚本
├── configs/                    # 配置文件
├── examples/                   # 示例数据
└── requirements.txt            # Python 依赖
```

---

## 📝 引用与致谢

本项目是对 **VideoVAE+** 开源工作的技术解读与延伸。  
若您使用本仓库内容，请引用原始论文并注明本仓库地址：

```bibtex
@inproceedings{xing2025videovaeplus,
  title={VideoVAE+: Large Motion Video Autoencoding with Cross-modal Video VAE},
  author={Xing, Yazhou and Fei, Yang and He, Yingqing and Chen, Jingye and Xie, Jiaxin and Chi, Xiaowei and Chen, Qifeng},
  booktitle={ICCV},
  year={2025}
}
```

同时，本仓库遵循原项目 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) 许可证。

---

## ⭐ Star History

如果这个项目对你有帮助，欢迎点个 Star！您的支持是我们持续分享的动力。
