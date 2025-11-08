"""
Training script for Vision Transformer model.

Trains a Vision Transformer (ViT) model for pneumonia detection.

Usage:
    python training/train_vit.py

The model settings can be configured in config/model_config.yaml.
Best model is automatically saved to checkpoints/vit/best_model.pth

Monitor training with TensorBoard:
    tensorboard --logdir logs
"""

import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.vit_model import create_vit_model
from data.dataloader import get_data_loaders, get_holdout_loader
from training.utils import (
    calculate_metrics,
    save_checkpoint,
    get_loss_function,
    EarlyStopping
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, log_interval=10):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Logging
        if batch_idx % log_interval == 0:
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = epoch_loss
    
    return metrics


def validate_epoch(model, val_loader, criterion, device, epoch, writer):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probas = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    metrics['loss'] = epoch_loss
    
    # Log metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f'Val/{key}', value, epoch)
    
    return metrics


def main():
    """Main training function."""
    # Load configurations
    with open('config/data_config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)['data']
    
    with open('config/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)['model']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_vit_model(model_config)
    model = model.to(device)
    print(f"Model created: Vision Transformer")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Data loaders
    train_loader, val_loader, test_loader = get_data_loaders(data_config, model_config)
    
    # Loss function
    criterion = get_loss_function(model_config, device)
    
    # Optimizer
    training_config = model_config['training']
    if training_config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config['learning_rate'],
            momentum=0.9,
            weight_decay=training_config['weight_decay']
        )
    
    # Learning rate scheduler
    scheduler_config = training_config.get('scheduler', 'cosine')
    if scheduler_config == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config['num_epochs'],
            eta_min=training_config.get('cosine_annealing', {}).get('eta_min', 1e-6)
        )
    elif scheduler_config == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_config.get('reduce_on_plateau', {}).get('factor', 0.5),
            patience=training_config.get('reduce_on_plateau', {}).get('patience', 5),
            min_lr=training_config.get('reduce_on_plateau', {}).get('min_lr', 1e-7)
        )
    else:
        scheduler = None
    
    # Early stopping
    early_stopping_config = training_config.get('early_stopping', {})
    early_stopping = EarlyStopping(
        patience=early_stopping_config.get('patience', 10),
        min_delta=early_stopping_config.get('min_delta', 0.001),
        monitor=early_stopping_config.get('monitor', 'val_loss'),
        mode='min' if 'loss' in early_stopping_config.get('monitor', 'val_loss') else 'max'
    )
    
    # Mixed precision
    use_amp = training_config.get('mixed_precision', True)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    # TensorBoard
    log_dir = Path(model_config['logging']['log_dir']) / 'vit'
    writer = SummaryWriter(log_dir)
    
    # Checkpoint directory
    checkpoint_dir = Path(model_config['checkpoint']['save_dir']) / 'vit'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    num_epochs = training_config['num_epochs']
    best_val_metric = float('-inf')
    monitor_metric = model_config['checkpoint']['monitor_metric']
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            log_interval=model_config['logging']['log_interval']
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch, writer)
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Log learning rate
        writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics.get('auc', 0.0):.4f}")
        
        # Checkpointing
        is_best = val_metrics[monitor_metric] > best_val_metric
        if is_best:
            best_val_metric = val_metrics[monitor_metric]
        
        if model_config['checkpoint']['save_best'] and is_best:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_dir / 'best_model.pth',
                is_best=True
            )
        
        if model_config['checkpoint']['save_last']:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_dir / 'last_model.pth',
                is_best=False
            )
        
        # Early stopping
        if early_stopping(val_metrics):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    writer.close()
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()


