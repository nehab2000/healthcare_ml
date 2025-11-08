# Getting Started Guide

A step-by-step guide for new users to get up and running quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU will work)
- At least 10GB free disk space for datasets

## Quick Setup (5 minutes)

### 1. Clone/Download the Project

```bash
cd Group_Project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Troubleshooting**: If you get errors, try:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare Your Data

You have two options:

#### Option A: Use Existing Datasets

1. Organize your datasets:
   ```
   data/
   ├── dataset1/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── dataset2/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── dataset3_holdout/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

2. Update `config/data_config.yaml` with your paths

3. Run data preparation:
   ```bash
   python scripts/prepare_data.py
   ```

#### Option B: Download NIH Dataset

1. Download the dataset:
   ```bash
   python scripts/download_nih_dataset.py --output_dir data/nih_dataset --cleanup
   ```

2. Organize images into NORMAL/PNEUMONIA folders (based on labels in Data_Entry.csv)

3. Update `config/data_config.yaml` with paths

4. Run data preparation:
   ```bash
   python scripts/prepare_data.py
   ```

#### Option C: Download Kaggle Dataset

1. Set up Kaggle API (if not already done):
   - Get credentials from: https://www.kaggle.com/settings
   - Place `kaggle.json` in `~/.kaggle/` directory

2. Download and organize the dataset:
   ```bash
   python scripts/download_kaggle_dataset.py --output_dir data/kaggle_dataset
   ```
   This automatically organizes images into NORMAL/PNEUMONIA folders.

3. Update `config/data_config.yaml` with the path to `data/kaggle_dataset/organized/`

4. Run data preparation:
   ```bash
   python scripts/prepare_data.py
   ```

### 4. Train Your First Model

Start with CNN (easier and faster):

```bash
python training/train_cnn.py
```

**What to expect:**
- Training will take 30 minutes to several hours depending on your GPU
- Progress bars will show training progress
- Best model is saved automatically to `checkpoints/cnn/best_model.pth`

**Monitor training** (optional):
```bash
# In a new terminal
tensorboard --logdir logs
# Open http://localhost:6006 in browser
```

### 5. Evaluate Your Model

```bash
python evaluation/evaluate.py \
    --model_type cnn \
    --checkpoint checkpoints/cnn/best_model.pth \
    --dataset test
```

## Common Workflows

### Workflow 1: Quick Test Run

Want to test if everything works? Use a small subset of data:

1. Create small test folders with ~100 images each
2. Run `prepare_data.py`
3. Train for 2-3 epochs (edit `num_epochs` in config)
4. Evaluate

### Workflow 2: Full Training Pipeline

1. **Prepare data** → `python scripts/prepare_data.py`
2. **Train CNN** → `python training/train_cnn.py`
3. **Train ViT** → `python training/train_vit.py`
4. **Compare results** → Check evaluation outputs
5. **Final evaluation** → Test on holdout set

### Workflow 3: Experiment with Different Models

1. Edit `config/model_config.yaml`:
   - Change `architecture` to try different CNNs
   - Adjust `learning_rate` and `batch_size`
   - Modify `num_epochs`

2. Train: `python training/train_cnn.py`

3. Compare results in TensorBoard or evaluation outputs

## Understanding the Output

### After `prepare_data.py`:

You'll see:
- Image verification report (check for issues)
- Duplicate detection report
- Dataset statistics (class distribution, split sizes)

**Check**: Make sure no major issues are reported!

### During Training:

You'll see:
- Epoch progress
- Train/Validation loss and accuracy
- Best model saved message

**Watch for**: 
- Loss decreasing
- Accuracy increasing
- Early stopping (if validation doesn't improve)

### After Evaluation:

You'll get:
- Detailed metrics (accuracy, F1, AUC, etc.)
- Confusion matrix
- ROC curve plot

**Key metrics**:
- **Accuracy**: Overall correctness
- **F1-Score**: Balanced metric
- **AUC**: Area under ROC curve (higher is better)

## Next Steps

Once you have a working model:

1. **Experiment**: Try different architectures
2. **Tune hyperparameters**: Adjust learning rate, batch size
3. **Visualize**: Use Grad-CAM to see what the model focuses on
4. **Compare**: Train both CNN and ViT, compare results
5. **Final test**: Evaluate on holdout set (only once!)

## Getting Help

- Check `README.md` for detailed documentation
- Review error messages carefully
- Verify your data structure
- Check configuration files

## Tips

- **Start small**: Test with a subset first
- **Save checkpoints**: Best models are auto-saved
- **Monitor training**: Use TensorBoard
- **Don't overfit**: Watch validation metrics
- **Use holdout wisely**: Only for final evaluation

Good luck! 🎉

