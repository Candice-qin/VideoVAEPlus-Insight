#!/bin/bash
# 请根据实际 Isaac Sim 安装路径修改
ISAAC_SIM_PATH=~/.local/share/ov/pkg/isaac_sim-2023.1.1
$ISAAC_SIM_PATH/isaac-sim.sh --ext-folder ./src/deployment --enable vae_extension
