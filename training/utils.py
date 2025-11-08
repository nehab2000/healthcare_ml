"""Training utilities: metrics, checkpointing, loss functions."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


def calculate_metrics(y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC)
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_normal': precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
        'recall_normal': recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
        'f1_normal': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        'precision_pneumonia': precision_per_class[1] if len(precision_per_class) > 1 else 0.0,
        'recall_pneumonia': recall_per_class[1] if len(recall_per_class) > 1 else 0.0,
        'f1_pneumonia': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
    }
    
    # AUC if probabilities provided
    if y_proba is not None:
        try:
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                # Multi-class: use probabilities for positive class
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba)
            metrics['auc'] = auc
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            metrics['auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict[str, float],
                   filepath: Path,
                   is_best: bool = False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        is_best: Whether this is the best model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = filepath.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)


def load_checkpoint(filepath: Path,
                   model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: str = 'cuda') -> Dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        device: Device to load on
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 monitor: str = 'val_loss',
                 mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            mode: 'min' or 'max' (minimize or maximize metric)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
        else:
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop.
        
        Args:
            metrics: Dictionary of current metrics
            
        Returns:
            True if should stop, False otherwise
        """
        if self.monitor not in metrics:
            print(f"Warning: Monitor metric '{self.monitor}' not found in metrics")
            return False
        
        current_score = metrics[self.monitor]
        
        if self.best_score is None or self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def get_loss_function(config: Dict, device: str = 'cuda') -> nn.Module:
    """
    Get loss function from configuration.
    
    Args:
        config: Model configuration dictionary
        device: Device to place weights on
        
    Returns:
        Loss function
    """
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'cross_entropy')
    
    if loss_type == 'cross_entropy':
        class_weights = loss_config.get('class_weights')
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()
    
    elif loss_type == 'focal_loss':
        # Focal Loss implementation
        alpha = loss_config.get('focal_alpha', 0.25)
        gamma = loss_config.get('focal_gamma', 2.0)
        
        class FocalLoss(nn.Module):
            def __init__(self, alpha=alpha, gamma=gamma):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            
            def forward(self, inputs, targets):
                ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


