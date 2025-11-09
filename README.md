<<<<<<< HEAD
# Pneumonia Detection using CNN and Vision Transformer

A comprehensive deep learning pipeline for detecting pneumonia from chest X-ray images using both Convolutional Neural Networks (CNN) and Vision Transformer (ViT) architectures.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Guide](#setup-guide)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

**For first-time users, follow these steps in order:**

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download datasets** (if using NIH dataset):
   ```bash
   python scripts/download_nih_dataset.py --output_dir data/nih_dataset --cleanup
   ```

3. **Organize your data** into the structure shown in [Setup Guide](#setup-guide)

4. **Configure paths** in `config/data_config.yaml`

5. **Prepare data:**
   ```bash
   python scripts/prepare_data.py
   ```

6. **Train a model:**
   ```bash
   # Train CNN
   python training/train_cnn.py
   
   # OR train Vision Transformer
   python training/train_vit.py
   ```

7. **Evaluate:**
   ```bash
   python evaluation/evaluate.py --model_type cnn --checkpoint checkpoints/cnn/best_model.pth --dataset test
   ```

## ✨ Features

- **Dual Architecture Support**: Train both CNN (ResNet50, DenseNet121, EfficientNet) and Vision Transformer models
- **Data Verification**: Automatic image verification and cleaning to ensure dataset consistency
- **Stratified Splitting**: Combines multiple datasets with proper train/val/test splits
- **Holdout Evaluation**: Separate holdout dataset for final model evaluation
- **Comprehensive Evaluation**: Detailed metrics, ROC curves, confusion matrices
- **Model Interpretability**: Grad-CAM for CNN and attention visualization for ViT
- **Reproducible**: Fixed random seeds and configuration-based setup

## 📁 Project Structure

```
Group_Project/
│
├── 📂 config/                          # Configuration files
│   ├── data_config.yaml               # Data paths, splits, verification settings
│   └── model_config.yaml              # Model architecture, training hyperparameters
│
├── 📂 data/                            # Data directories (create these)
│   ├── dataset1/                      # First dataset to combine
│   │   ├── NORMAL/                    # Normal X-ray images
│   │   └── PNEUMONIA/                 # Pneumonia X-ray images
│   ├── dataset2/                      # Second dataset to combine
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── dataset3_holdout/              # Holdout dataset (for final evaluation only)
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── combined/                      # ⚠️ Created by prepare_data.py
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── holdout/                       # ⚠️ Created by prepare_data.py
│
├── 📂 scripts/                         # Main execution scripts
│   ├── download_nih_dataset.py       # Download NIH Chest X-ray dataset
│   ├── download_kaggle_dataset.py    # Download Kaggle Chest X-ray Pneumonia dataset
│   └── prepare_data.py               # ⭐ Start here: Prepare and combine datasets
│
├── 📂 data_preparation/                # Data processing modules
│   ├── combine_datasets.py           # Combines datasets with stratified splitting
│   ├── verify_images.py              # Verifies image quality, size, format
│   └── detect_duplicates.py          # Finds duplicate images across datasets
│
├── 📂 models/                          # Model architectures
│   ├── cnn_model.py                  # CNN models (ResNet, DenseNet, EfficientNet)
│   ├── vit_model.py                  # Vision Transformer model
│   └── base_model.py                 # Shared model utilities
│
├── 📂 data/                            # Data loading utilities
│   └── dataloader.py                  # PyTorch Dataset and DataLoader
│
├── 📂 training/                        # Training scripts
│   ├── train_cnn.py                   # ⭐ Train CNN model
│   ├── train_vit.py                   # ⭐ Train Vision Transformer
│   └── utils.py                       # Training utilities (metrics, checkpointing)
│
├── 📂 evaluation/                      # Evaluation and visualization
│   ├── evaluate.py                    # ⭐ Evaluate trained models
│   └── visualize.py                   # Grad-CAM and attention visualizations
│
├── 📂 checkpoints/                     # ⚠️ Created during training
│   ├── cnn/
│   └── vit/
│
├── 📂 logs/                            # ⚠️ Created during training (TensorBoard)
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

**Legend:**
- ⭐ = Main entry points for common tasks
- ⚠️ = Directories/files created automatically (don't create manually)

## 📖 Setup Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download Datasets (Optional)

**Option A: NIH Chest X-ray Dataset**
```bash
# Download, extract, and cleanup
python scripts/download_nih_dataset.py --output_dir data/nih_dataset --cleanup
```
**Note**: After downloading, you'll need to organize images into NORMAL/PNEUMONIA folders based on labels in Data_Entry.csv.

**Option B: Kaggle Chest X-ray Pneumonia Dataset**
```bash
# Download and organize automatically
python scripts/download_kaggle_dataset.py --output_dir data/kaggle_dataset
```
**Note**: For Kaggle, you may need to set up API credentials:
1. Get API credentials from: https://www.kaggle.com/settings
2. Place `kaggle.json` in `~/.kaggle/` directory (or set `KAGGLE_CONFIG_DIR` environment variable)
3. The script automatically organizes images into NORMAL/PNEUMONIA folders

### Step 3: Organize Your Data

Create the following directory structure:

```
data/
├── dataset1/
│   ├── NORMAL/
│   │   └── (normal x-ray images)
│   └── PNEUMONIA/
│       └── (pneumonia x-ray images)
├── dataset2/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── dataset3_holdout/
    ├── NORMAL/
    └── PNEUMONIA/
```

**Important**: 
- `dataset1` and `dataset2` will be **combined** for training
- `dataset3_holdout` will be kept **separate** for final evaluation only

### Step 4: Configure Settings

Edit `config/data_config.yaml`:

```yaml
data:
  dataset1_path: "data/dataset1"           # Change to your path
  dataset2_path: "data/dataset2"           # Change to your path
  dataset3_holdout_path: "data/dataset3_holdout"  # Change to your path
  
  train_ratio: 0.7    # 70% for training
  val_ratio: 0.15     # 15% for validation
  test_ratio: 0.15    # 15% for testing
```

Edit `config/model_config.yaml` to adjust:
- Model architecture (ResNet50, EfficientNet, etc.)
- Training hyperparameters (learning rate, batch size, epochs)
- Data augmentation settings

## 📚 Usage Guide

### Workflow Overview

```
1. Prepare Data → 2. Train Model → 3. Evaluate → 4. Visualize
```

### Step 1: Prepare Data

This step verifies images, detects duplicates, and creates train/val/test splits:

```bash
python scripts/prepare_data.py
```

**What it does:**
- ✅ Verifies all images (size, format, quality)
- ✅ Detects duplicate images across datasets
- ✅ Combines dataset1 and dataset2
- ✅ Creates stratified train/val/test splits (maintains class balance)
- ✅ Copies holdout dataset separately
- ✅ Generates statistics report

**Output:**
- `data/combined/train/` - Training images
- `data/combined/val/` - Validation images
- `data/combined/test/` - Test images
- `data/holdout/` - Holdout images

### Step 2: Train Models

**Train CNN model:**
```bash
python training/train_cnn.py
```

**Train Vision Transformer:**
```bash
python training/train_vit.py
```

**Monitor training:**
```bash
# In a separate terminal
tensorboard --logdir logs
# Then open http://localhost:6006 in your browser
```

**Output:**
- `checkpoints/cnn/best_model.pth` - Best CNN model
- `checkpoints/vit/best_model.pth` - Best ViT model
- `logs/cnn/` - Training logs for CNN
- `logs/vit/` - Training logs for ViT

### Step 3: Evaluate Models

**Evaluate on test set:**
```bash
python evaluation/evaluate.py \
    --model_type cnn \
    --checkpoint checkpoints/cnn/best_model.pth \
    --dataset test \
    --output_dir evaluation_results
```

**Evaluate on holdout set (final evaluation):**
```bash
python evaluation/evaluate.py \
    --model_type cnn \
    --checkpoint checkpoints/cnn/best_model.pth \
    --dataset holdout \
    --output_dir evaluation_results
```

**Output:**
- `evaluation_results/cnn_test_results.json` - Metrics
- `evaluation_results/cnn_test_roc.png` - ROC curve
- `evaluation_results/cnn_test_cm.png` - Confusion matrix

### Step 4: Visualize Results

See `evaluation/visualize.py` for visualization functions:
- Sample predictions with confidence scores
- Grad-CAM heatmaps (CNN)
- Attention maps (ViT)

## ⚙️ Configuration

### Data Configuration (`config/data_config.yaml`)

Key settings:
- **Dataset paths**: Where your datasets are located
- **Split ratios**: How to divide train/val/test (must sum to 1.0)
- **Image verification**: Target size, allowed formats, quality checks
- **Batch size**: Images per batch (adjust based on GPU memory)

### Model Configuration (`config/model_config.yaml`)

Key settings:
- **Architecture**: Choose CNN (resnet50, densenet121, efficientnet_b3) or ViT
- **Learning rate**: Start with 1e-4 for fine-tuning
- **Batch size**: 32 is a good starting point
- **Epochs**: 50 with early stopping (patience=10)
- **Optimizer**: AdamW (recommended) or SGD
- **Scheduler**: Cosine annealing (recommended)

## 🏗️ Model Architectures

### CNN Models

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| ResNet50 | ~25M | Fast | High | Good starting point |
| ResNet101 | ~44M | Medium | Very High | More capacity |
| DenseNet121 | ~8M | Fast | High | Efficient |
| EfficientNet-B3 | ~12M | Medium | Very High | Best accuracy/speed |

### Vision Transformer

- **ViT-Base**: 12 layers, 768 hidden size
- Pretrained on ImageNet
- May require more data than CNN
- Good for comparison

## 📊 Evaluation Metrics

The evaluation module provides:

- **Overall Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
- **Per-Class Metrics**: Separate metrics for NORMAL and PNEUMONIA
- **Confusion Matrix**: True/False positives and negatives
- **ROC Curve**: Visual representation of model performance

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory Error**
- **Solution**: Reduce `batch_size` in `config/model_config.yaml`
- Try: `batch_size: 16` or `batch_size: 8`

**2. Slow Training**
- **Solution**: Mixed precision is already enabled by default
- Consider using a smaller model (ResNet50 instead of ResNet101)

**3. Poor Model Performance**
- Check data quality: Run `prepare_data.py` and review verification report
- Try different architectures
- Adjust learning rate (try 1e-5 or 5e-5)
- Increase training epochs
- Check class balance in statistics

**4. Image Verification Failures**
- Check image formats (should be JPEG or PNG)
- Verify image sizes (will be resized to 224x224 automatically)
- Review verification report for specific issues

**5. Import Errors**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)

**6. Dataset Not Found**
- Verify paths in `config/data_config.yaml`
- Ensure directory structure matches expected format
- Check that NORMAL and PNEUMONIA folders exist

## 📝 File Descriptions

### Main Scripts

- **`scripts/prepare_data.py`**: First script to run - prepares and combines datasets
- **`scripts/download_nih_dataset.py`**: Downloads NIH Chest X-ray dataset
- **`training/train_cnn.py`**: Trains CNN models
- **`training/train_vit.py`**: Trains Vision Transformer
- **`evaluation/evaluate.py`**: Evaluates trained models

### Configuration Files

- **`config/data_config.yaml`**: Data paths, splits, verification settings
- **`config/model_config.yaml`**: Model architecture and training hyperparameters

### Key Modules

- **`data_preparation/`**: All data processing and verification
- **`models/`**: Model architectures
- **`training/utils.py`**: Training utilities (metrics, checkpointing, early stopping)
- **`evaluation/`**: Evaluation and visualization tools

## 🎯 Expected Performance

Typical performance ranges (may vary based on dataset):

- **CNN (ResNet50)**: 90-94% accuracy
- **EfficientNet**: 92-95% accuracy  
- **ViT**: 91-94% accuracy
- **Ensemble**: 94-96% accuracy

## 💡 Tips for Group Members

1. **Start Simple**: Begin with ResNet50 CNN - it's fast and reliable
2. **Monitor Training**: Use TensorBoard to watch training progress
3. **Save Checkpoints**: Best models are automatically saved
4. **Use Holdout Set Sparingly**: Only for final evaluation, not during development
5. **Check Data Quality**: Always review the verification report from `prepare_data.py`
6. **Experiment**: Try different architectures and hyperparameters
7. **Document Changes**: Note what works and what doesn't

## 📞 Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review error messages carefully
3. Verify your data structure matches the expected format
4. Check configuration files for typos
5. Ensure all dependencies are installed

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- NIH Chest X-ray dataset
- PyTorch and torchvision
- HuggingFace Transformers
- Albumentations for data augmentation

---

**Happy Training! 🚀**
=======
