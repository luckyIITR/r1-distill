#!/usr/bin/env bash
set -e

# Detect max CUDA version supported by the driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "No GPU detected, installing CPU torch"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    exit 0
fi

CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | awk -F. '{print $1}')

# Map driver major version to a torch CUDA build
# Driver >= 555 supports CUDA 12.6+; >= 560 supports 12.8; >= 570 supports 12.9+
if nvidia-smi | grep -q "CUDA Version: 13"; then
    INDEX="https://download.pytorch.org/whl/cu130"
elif nvidia-smi | grep -q "CUDA Version: 12.9\|CUDA Version: 12.8"; then
    INDEX="https://download.pytorch.org/whl/cu128"
elif nvidia-smi | grep -q "CUDA Version: 12.6\|CUDA Version: 12.7"; then
    INDEX="https://download.pytorch.org/whl/cu126"
elif nvidia-smi | grep -q "CUDA Version: 12"; then
    INDEX="https://download.pytorch.org/whl/cu124"
else
    INDEX="https://download.pytorch.org/whl/cu121"
fi

echo "Installing torch from $INDEX"
uv pip install --reinstall torch torchvision --index-url "$INDEX"