# YOLO-World_EfficientViT-SAM

## Getting Started

[HuggingFace Space](https://huggingface.co/spaces/pg56714/YOLO-World_EfficientViT-SAM)

![example1](/assets/example1.jpg)

## Installation

download the pretrained weights from the following links and save them in the `weights` directory.
https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl1.pt (save it as `weights/xl1.pt`)

Use Anaconda to create a new environment and install the required packages.

```bash
uv venv

.venv\Scripts\activate

uv pip install -r pyproject.toml
```

PyTorch is installed from the CUDA 12.6 wheel index (`pytorch-cu126` in `pyproject.toml`), which needs an NVIDIA driver that supports CUDA 12.6 or newer.

## Running the Project

```bash
uv run app.py
```
