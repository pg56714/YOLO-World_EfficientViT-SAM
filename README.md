# YOLO-World_EfficientViT-SAM

## Getting Started

[HuggingFace Space](https://huggingface.co/spaces/pg56714/YOLO-World_EfficientViT-SAM)

![example1](/assets/example1.jpg)

## Installation

download the pretrained weights from the following links and save them in the `weights` directory.
https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt

Use Anaconda to create a new environment and install the required packages.

```bash
uv venv

.venv\Scripts\activate

uv pip install -r pyproject.toml
```

## Running the Project

```bash
uv run app.py
```
