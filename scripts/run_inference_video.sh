#!/bin/bash
python -m src.inference_video \
    --data_root 'examples/videos/gt' \
    --out_root 'examples/videos/recon' \
    --config_path 'configs/config_16z.yaml' \
    --chunk_size 8 \
    --resolution 720 1280
