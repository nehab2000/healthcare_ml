# Quick Reference Card

## 🚀 Most Common Commands

```bash
# 0. Download datasets (if needed)
python scripts/download_nih_dataset.py --output_dir data/nih_dataset --cleanup
python scripts/download_kaggle_dataset.py --output_dir data/kaggle_dataset

# 1. Prepare data (FIRST STEP)
python scripts/prepare_data.py

# 2. Train CNN
python training/train_cnn.py

# 3. Train ViT
python training/train_vit.py

# 4. Evaluate model
python evaluation/evaluate.py --model_type cnn --checkpoint checkpoints/cnn/best_model.pth --dataset test

# 5. Monitor training
tensorboard --logdir logs
```

## 📁 Where Things Are

| What | Where |
|------|-------|
| **Your datasets** | `data/dataset1/`, `data/dataset2/`, `data/dataset3_holdout/` |
| **Config files** | `config/data_config.yaml`, `config/model_config.yaml` |
| **Main scripts** | `scripts/` folder |
| **Trained models** | `checkpoints/cnn/`, `checkpoints/vit/` |
| **Training logs** | `logs/` folder (view with TensorBoard) |
| **Results** | `evaluation_results/` (created after evaluation) |

## ⚙️ Configuration Quick Edits

### Change dataset paths
Edit `config/data_config.yaml`:
```yaml
data:
  dataset1_path: "data/your_dataset1"
  dataset2_path: "data/your_dataset2"
```

### Change model architecture
Edit `config/model_config.yaml`:
```yaml
model:
  cnn:
    architecture: "resnet50"  # or "densenet121", "efficientnet_b3"
```

### Adjust batch size (if out of memory)
Edit `config/model_config.yaml`:
```yaml
model:
  training:
    batch_size: 16  # Reduce from 32 if needed
```

## 📊 Workflow Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Organize datasets in `data/` folder
- [ ] Update `config/data_config.yaml` with paths
- [ ] Run `python scripts/prepare_data.py`
- [ ] Check verification report for issues
- [ ] Train model: `python training/train_cnn.py`
- [ ] Evaluate: `python evaluation/evaluate.py ...`

## 🎯 File Purposes

| File | Purpose |
|------|---------|
| `scripts/prepare_data.py` | ⭐ **START HERE** - Prepares datasets |
| `training/train_cnn.py` | Train CNN models |
| `training/train_vit.py` | Train Vision Transformer |
| `evaluation/evaluate.py` | Evaluate trained models |
| `scripts/download_nih_dataset.py` | Download NIH dataset |
| `scripts/download_kaggle_dataset.py` | Download Kaggle dataset |

## 💡 Pro Tips

1. **Always run `prepare_data.py` first** - It verifies and organizes your data
2. **Check TensorBoard** - Monitor training progress visually
3. **Save space** - Use `--cleanup` when downloading datasets
4. **Start small** - Test with a subset of data first
5. **Holdout set** - Only use for final evaluation, not during development

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size` in config |
| Can't find dataset | Check paths in `config/data_config.yaml` |
| Import errors | Run `pip install -r requirements.txt` |
| Slow training | Already optimized, but check GPU usage |

## 📚 Documentation Files

- **`README.md`** - Full documentation
- **`GETTING_STARTED.md`** - Step-by-step guide for beginners
- **`PROJECT_STRUCTURE.md`** - Detailed structure explanation
- **`QUICK_REFERENCE.md`** - This file

---

**Need help?** Check the full README.md or GETTING_STARTED.md

