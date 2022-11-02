# rs-opencv
## Introduction

Using:

- Ubuntu 20.04 (Requirement) I think, it is supported in Ubuntu 18.04 has to be test
- Python 3.9 (Requirement)
- TensorRT
- Pytorch
- OpenCV
- RealSense Camera (D435)

⚡ The script is working. It need to improve in the part of handle the recognized objects. Also the recognition model has to be improve.

⚡ Support for regular cameras has to be included.

## Requirements

- Ubuntu 20.04 or Ubuntu 18.04
- **Python3.9**
- **CUDA cores** (NVIDIA GPU or Jetson Nano)

### Test if CUDA is available

```bash
python3.9
>>import torch
>>torch.cuda.is_available()
True
```

**If result is False:**

Check: [After Installing Ubuntu 20.04](https://www.notion.so/After-Installing-Ubuntu-20-04-9df3a7798a2a41ff8e547fa79b25d61a) 

## Setup

```bash
git clone https://github.com/john-wick1999/rs-opencv.git
cd rs-opencv/
virtualenv -p /bin/python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Start

### Activate virtual environment

```bash
source venv/bin/activate
```

### Add models

Models are located in Data folder with name (new model change best.engine):

- best.engine
- coco.yaml

### Run script

Activate venv.

Connect camera.

```bash
python detect.py
```
