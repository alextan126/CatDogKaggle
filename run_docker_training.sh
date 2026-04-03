#!/bin/bash
# Run the training script inside an NVIDIA container with full GB10 (Blackwell) support

# The container needs to mount the current directory so it can read the code and save artifacts
WORKSPACE_DIR=$(pwd)

echo "Starting training in NVIDIA PyTorch Docker container..."
docker run --gpus all --rm -it \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WORKSPACE_DIR:/workspace" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.04-py3 \
    bash -c "pip install -q -r requirements.txt && python main.py"
