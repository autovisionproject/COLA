## [ECCV-2026] COLA: Continual Orthogonal Low-Rank Adaptation for Class-Incremental Learning
We provide the complete codebase used to benchmark continual learning performance on four popular benchmark datasets for CL.

### 1. Download Datasets
We conduct our experiments on the following datasets:
- ImageNet-A, ImageNet-R, CUB200, DomainNet
- [Download dataset: [Reff-Link](https://github.com/LAMDA-CL/LAMDA-PILOT)]
```
Update dataset path : utils/data.py
 e.g.,
  - train_dir = "./data/imagenet-r/train/" 
  - test_dir = "./data/imagenet-r/test/"
```
### 2. System Configuration
```
Python 3.12.3
Ubuntu 24.04.1 LTS
CUDA Version: 12.4
```
### 3. Train Model
```
Model Details: [ViT-B/16, DeiT-B/16, DeiT-S/16]
Datset Details: [ImageNet-A, ImageNet-R, CUB200, DomainNet]

======== Run CoLa ================
#python3 main.py --config=./exps/cola_<config-file>.json

========= Run Baseline ====================
#python3 main.py --config=./exps/<baseline-config-file>.json

```
## Citation
* Please cite this work if you find it useful for your research.
```
```

## Acknowledgement
This repo is built upon the following project. We sincerely appreciate for their contributions.

1. [SD-Lora-CL](https://github.com/WuYichen-97/SD-Lora-CL)  

