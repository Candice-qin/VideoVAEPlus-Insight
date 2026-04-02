#!/bin/bash
python -m src.inference_image \
    --data_root 'examples/images/gt' \
    --out_root 'examples/images/recon' \
    --config_path 'configs/config_16z.yaml' \
    --batch_size 1
