#!/bin/bash
WORKSPACE_DIR=$(pwd)

echo "Generating plots using saved weights in NVIDIA PyTorch Docker container..."
docker run --gpus all --rm -it \
    --shm-size=8g \
    -v "$WORKSPACE_DIR:/workspace" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.04-py3 \
    bash -c "pip install -q -r requirements.txt && python generate_plots.py"
