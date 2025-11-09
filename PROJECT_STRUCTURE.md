# Project Structure Guide

A detailed guide to understanding the project organization.

## Directory Overview

```
Group_Project/
│
├── 📂 config/              → Configuration files (edit these)
├── 📂 data/                → Your datasets (create these)
├── 📂 scripts/             → Main scripts to run
├── 📂 data_preparation/    → Data processing code
├── 📂 models/              → Model architectures
├── 📂 training/            → Training code
├── 📂 evaluation/          → Evaluation code
├── 📂 checkpoints/         → Saved models (auto-created)
└── 📂 logs/                → Training logs (auto-created)
```

## Detailed Structure

### 📂 config/

**Purpose**: All configuration settings

**Files**:
- `data_config.yaml` - Data paths, split ratios, verification settings
- `model_config.yaml` - Model architecture, training hyperparameters

**When to edit**: 
- Before running `prepare_data.py` (data paths)
- Before training (hyperparameters)

---

### 📂 data/

**Purpose**: Store your datasets

**Structure**:
```
data/
├── dataset1/              # First dataset (will be combined)
│   ├── NORMAL/
│   └── PNEUMONIA/
├── dataset2/              # Second dataset (will be combined)
│   ├── NORMAL/
│   └── PNEUMONIA/
├── dataset3_holdout/      # Holdout dataset (kept separate)
│   ├── NORMAL/
│   └── PNEUMONIA/
├── combined/             # ⚠️ Created by prepare_data.py
│   ├── train/
│   ├── val/
│   └── test/
└── holdout/              # ⚠️ Created by prepare_data.py
```

**When to create**: Before running `prepare_data.py`

---

### 📂 scripts/

**Purpose**: Main entry point scripts

**Files**:
- `download_nih_dataset.py` - Downloads NIH Chest X-ray dataset
- `download_kaggle_dataset.py` - Downloads Kaggle Chest X-ray Pneumonia dataset
- `prepare_data.py` - **⭐ START HERE** - Prepares and combines datasets

**When to run**:
- `download_nih_dataset.py` - If using NIH dataset
- `download_kaggle_dataset.py` - If using Kaggle dataset
- `prepare_data.py` - First step in workflow (after organizing data)

---

### 📂 data_preparation/

**Purpose**: Data processing modules (you don't run these directly)

**Files**:
- `combine_datasets.py` - Combines datasets with stratified splitting
- `verify_images.py` - Verifies image quality and consistency
- `detect_duplicates.py` - Finds duplicate images

**Used by**: `scripts/prepare_data.py`

---

### 📂 models/

**Purpose**: Model architecture definitions

**Files**:
- `cnn_model.py` - CNN architectures (ResNet, DenseNet, EfficientNet)
- `vit_model.py` - Vision Transformer
- `base_model.py` - Shared utilities

**When to edit**: If you want to modify model architectures

---

### 📂 data/

**Purpose**: Data loading utilities

**Files**:
- `dataloader.py` - PyTorch Dataset and DataLoader classes

**Used by**: Training scripts

---

### 📂 training/

**Purpose**: Training scripts and utilities

**Files**:
- `train_cnn.py` - **⭐ Train CNN models**
- `train_vit.py` - **⭐ Train Vision Transformer**
- `utils.py` - Training utilities (metrics, checkpointing, early stopping)

**When to run**: After data preparation

---

### 📂 evaluation/

**Purpose**: Model evaluation and visualization

**Files**:
- `evaluate.py` - **⭐ Evaluate trained models**
- `visualize.py` - Visualization tools (Grad-CAM, attention maps)

**When to run**: After training

---

### 📂 checkpoints/ (auto-created)

**Purpose**: Saved model checkpoints

**Structure**:
```
checkpoints/
├── cnn/
│   ├── best_model.pth    # Best CNN model
│   └── last_model.pth    # Last epoch model
└── vit/
    ├── best_model.pth    # Best ViT model
    └── last_model.pth    # Last epoch model
```

**Created by**: Training scripts

---

### 📂 logs/ (auto-created)

**Purpose**: TensorBoard training logs

**Structure**:
```
logs/
├── cnn/                  # CNN training logs
└── vit/                  # ViT training logs
```

**View with**: `tensorboard --logdir logs`

---

## File Flow Diagram

```
1. User organizes data
   ↓
2. scripts/prepare_data.py
   ↓ (uses)
   data_preparation/combine_datasets.py
   data_preparation/verify_images.py
   data_preparation/detect_duplicates.py
   ↓ (creates)
   data/combined/ (train/val/test)
   data/holdout/
   ↓
3. training/train_cnn.py OR training/train_vit.py
   ↓ (uses)
   models/cnn_model.py OR models/vit_model.py
   data/dataloader.py
   training/utils.py
   ↓ (creates)
   checkpoints/cnn/ OR checkpoints/vit/
   logs/cnn/ OR logs/vit/
   ↓
4. evaluation/evaluate.py
   ↓ (uses)
   checkpoints/cnn/best_model.pth OR checkpoints/vit/best_model.pth
   data/combined/test/ OR data/holdout/
   ↓ (creates)
   evaluation_results/
```

## Key Files to Know

### For Data Preparation
- **`scripts/prepare_data.py`** - Main script to run
- **`config/data_config.yaml`** - Configure paths and settings

### For Training
- **`training/train_cnn.py`** - Train CNN
- **`training/train_vit.py`** - Train ViT
- **`config/model_config.yaml`** - Configure hyperparameters

### For Evaluation
- **`evaluation/evaluate.py`** - Evaluate models
- **`checkpoints/cnn/best_model.pth`** - Best CNN model
- **`checkpoints/vit/best_model.pth`** - Best ViT model

## What You Need to Create

**Before starting**:
- `data/dataset1/` with NORMAL/ and PNEUMONIA/ folders
- `data/dataset2/` with NORMAL/ and PNEUMONIA/ folders
- `data/dataset3_holdout/` with NORMAL/ and PNEUMONIA/ folders

**Everything else is created automatically**:
- `data/combined/` - Created by `prepare_data.py`
- `data/holdout/` - Created by `prepare_data.py`
- `checkpoints/` - Created during training
- `logs/` - Created during training
- `evaluation_results/` - Created during evaluation

## Navigation Tips

1. **Start with scripts/** - These are the main entry points
2. **Check config/** - All settings are here
3. **Look at data/** - Your datasets go here
4. **Results in checkpoints/** and **logs/** - Created automatically

## Quick Reference

| Task | File to Run | Config to Edit |
|------|-------------|----------------|
| Prepare data | `scripts/prepare_data.py` | `config/data_config.yaml` |
| Train CNN | `training/train_cnn.py` | `config/model_config.yaml` |
| Train ViT | `training/train_vit.py` | `config/model_config.yaml` |
| Evaluate | `evaluation/evaluate.py` | None (uses command line args) |

