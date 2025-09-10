# SAR-RARP50 Challenge – Semantic Segmentation

## Overview
This repository provides implementations for semantic segmentation on the **SAR-RARP50 2022 Challenge**, focusing on generating segmentation masks for surgical tool parts and small/thin objects such as surgical clips, suturing threads, and needles.  

We implement and compare three algorithms:

1. **DeepLab** (with ResNet and MobileNet backbones)  
2. **SegFormer** (with a combined loss function, including `BoundaryLoss`, to improve thin object detection such as threads)  
3. **SAM2 Semantic**  

Initial results show that **SegFormer** performs best for this task.  

---

## Dataset
- **Train set**: [SAR-RARP50 Train Set](https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_train_set/24932529)  
- **Test set**: [SAR-RARP50 Test Set](https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_test_set/24932499)  

---

## Development Environment
- **OS**: Ubuntu 24.04.2 LTS  
- **Python**: 3.12.3  
- **GPU**: NVIDIA GeForce RTX 4060 (8 GB memory)  

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/bakhshandeh/SAR-RARP50-Challenge.git
cd SAR-RARP50-Challenge
```

### 2. Setup virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Download and prepare dataset
Create dataset folders:
```bash
mkdir -p dataset/train dataset/test
```

- Download the **train** and **test** datasets from the links above.  
- Place the `.zip` files into `dataset/train` and `dataset/test`.  
- Expected structure before extraction:
  ```
  dataset/train
  ├── video_01.zip
  ├── video_02.zip
  ...

  dataset/test
  ├── video_40.zip
  ├── video_41.zip
  ...
  ```

Run the dataset preparation script:
```bash
chmod +x prepare_dataset.sh
./prepare_dataset.sh
```
This script:
- Unzips the video archives.  
- Extracts frames into `/rgb` folders.  
- Skips unlabeled frames, keeping only those with segmentation labels.  

### 4. Install requirements
```bash
pip install -r requirements.txt
```

---

## Training

### DeepLab
```bash
python train_deeplab.py --root dataset --batch_size 2 --size 512x288 --disable_aux --backbone mobilenet --bn freeze
```

### SegFormer
```bash
python segformer_train.py --root dataset --epochs 40 --batch_size 2 --lr 6e-5 --wd 1e-4 --size 1024x576 --variant b2 --progress bar
```

### SAM2 Semantic
Install additional dependencies:
```bash
pip install -r requirements-sam2.txt
```

Train:
```bash
python train_sam2_semantic.py --root dataset --size 896x512 --batch_size 2 --epochs 40 --lr 6e-5 --wd 1e-4 --freeze_backbone --fallback_variant vit_b --num_workers 4
```

---

## Inference

Example using **SegFormer**:
```bash
python inference_segformer.py   --frame dataset/train/video_01/rgb/000000000.png   --checkpoint checkpoints_segformer/best_segformer.pth   --variant b2   --size 1024x576
```

The masked output will be saved, and its file path printed.  
(Similar commands apply for DeepLab and SAM2 Semantic.)

---

## Next Steps
- [ ] Dockerize the project for easier deployment  
- [ ] Optimize loss function (e.g., adjusting `BoundaryLoss` weights for improved thin-object detection)  
- [ ] Full training/testing (current results limited by compute resources)  
