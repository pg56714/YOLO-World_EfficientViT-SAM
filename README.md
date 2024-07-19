# YOLO-World_EfficientViT-SAM

## Getting Started

[HuggingFace Space](https://huggingface.co/spaces/pg56714/YOLO-World_EfficientViT-SAM)

![example1](/assets/example1.jpg)

## Installation

download the pretrained weights from the following links and save them in the `weights` directory.
https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt

Use Anaconda to create a new environment and install the required packages.

```bash
# Create and activate a python 3.10 environment.
conda create -n yolo-world-efficientvit-sam python=3.10 -y

conda activate yolo-world-efficientvit-sam

pip install -r requirements.txt
```

## Running the Project

```bash
python app.py
```

## Core Models

### YOLO-World

[YOLO-World](https://github.com/AILab-CVC/YOLO-World) is an open-vocabulary object detection model with high efficiency.
On the challenging LVIS dataset, YOLO-World achieves 35.4 AP with 52.0 FPS on V100,
which outperforms many state-of-the-art methods in terms of both accuracy and speed.

<img width="1024" src="https://github.com/Curt-Park/yolo-world-with-efficientvit-sam/assets/14961526/8a4a17bd-918d-478a-8451-f58e4a2dce79">
<img width="1024" src="https://github.com/Curt-Park/yolo-world-with-efficientvit-sam/assets/14961526/fce57405-e18d-45f3-bea8-fc3971faf975">

### EfficientViT-SAM

[EfficientViT SAM](https://github.com/mit-han-lab/efficientvit) is a new family of accelerated segment anything models.
Thanks to the lightweight and hardware-efficient core building block,
it delivers 48.9× measured TensorRT speedup on A100 GPU over SAM-ViT-H without sacrificing performance.

<img width="1024" src="https://github.com/Curt-Park/yolo-world-with-efficientvit-sam/assets/14961526/9eec003f-47c9-43a5-86b0-82d6689e1bf9">
<img width="1024" src="https://github.com/Curt-Park/yolo-world-with-efficientvit-sam/assets/14961526/d79973bb-0d80-4b64-a175-252de56d0d09">

## Citation

```
@misc{zhang2024efficientvitsam,
  title={EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss},
  author={Zhuoyang Zhang and Han Cai and Song Han},
  year={2024},
  eprint={2402.05008},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@article{cheng2024yolow,
  title={YOLO-World: Real-Time Open-Vocabulary Object Detection},
  author={Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying},
  journal={arXiv preprint arXiv:2401.17270},
  year={2024}
}
```
