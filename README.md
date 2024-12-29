# Fine-tuning Methods Comparison for RAD-DINO Models in Medical XRays Images

This repository contains research on comparing three different fine-tuning methods applied to RAD-DINO and RAD-DINO-MAIRA-2 encoder models.

## Overview

We investigate the effectiveness of three distinct fine-tuning approaches:
1. Linear Probing
2. Full Fine-tuning
3. LoRA (Low-Rank Adaptation)

The study was conducted using two base models:
- [RAD-DINO](https://huggingface.co/microsoft/rad-dino)
- [RAD-DINO-MAIRA-2](https://huggingface.co/microsoft/rad-dino-maira-2)

Dataset:
- [VinDigData](https://www.kaggle.com/code/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-train/data) 

## How to use

1. Install all required packages

```bash
pip install -r requirements.txt
```

2. Prepare dataset (see data [README](https://github.com/tlidzhiev/ssl-in-medical-xrays-images/tree/main/data))
```bash
python data/generate_dataset.py
```

3. Adjust the parameters and train necessary `train` file:
 
```bash
python train_....py
```


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

