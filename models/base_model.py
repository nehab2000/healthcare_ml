"""Base model utilities and common functions."""

import torch
import torch.nn as nn
from typing import Optional


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        
    Returns:
        Summary string
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
Model Summary:
  Total parameters: {total_params:,}
  Trainable parameters: {trainable_params:,}
  Non-trainable parameters: {total_params - trainable_params:,}
  
Architecture:
{model}
"""
    return summary


def initialize_weights(module: nn.Module):
    """Initialize weights for custom layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


