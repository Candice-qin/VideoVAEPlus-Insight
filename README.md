# VideoVAEPlus-Insight
本项目是对 **[VideoVAE+](https://github.com/VideoVerses/VideoVAEPlus)** (ICCV 2025) 的深度技术解读、二次开发与具身智能部署实践。   > 我们不仅复现了原模型的训练与推理，更深入剖析了其核心设计（时空分离编码、文本引导、多尺度KL等），并进一步将潜空间与 **CoWVLA** 结合，在 **Isaac Sim** 中完成了机器人闭环控制的全链路实现。
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

- [背景介绍](#背景介绍)
- [仓库内容概览](#仓库内容概览)
- [快速开始](#快速开始)
- [技术深度解读](#技术深度解读)
- [具身智能部署](#具身智能部署)
- [引用与致谢](#引用与致谢)
- [许可证](#许可证)

---

## 背景介绍

VideoVAE+ 是首个专门针对**大幅运动视频**的高保真 Video VAE，通过时空分离编码和跨模态文本引导，大幅提升了重建质量。  
然而，原项目主要聚焦在视频压缩与重建本身，未深入探讨其在**具身智能**（尤其是 VLA 模型）中的应用潜力。  
本项目旨在填补这一空白：

1. 完整还原 VideoVAE+ 的训练与推理流程
2. 深度解析其核心模块设计原理
3. 将其潜空间与 CoWVLA 结合，构建世界模型推理链
4. 在 Isaac Sim 中完成真机仿真部署

---

## 仓库内容概览

| 目录/文件 | 说明 |
|-----------|------|
| `docs/` | 各模块详细技术文档（含代码片段与图解） |
| `src/` | 可复用的模型实现、训练/评估/部署脚本 |
| `scripts/` | 一键运行脚本 |
| `configs/` | 配置文件模板 |
| `examples/` | 示例数据 |

---

## 快速开始

### 环境配置

```bash
git clone https://github.com/yourname/VideoVAEPlus-Insight.git
cd VideoVAEPlus-Insight
conda create --name vae_insight python=3.10 -y
conda activate vae_insight
pip install -r requirements.txt
