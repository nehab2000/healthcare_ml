"""Model architectures for pneumonia detection."""

from .cnn_model import PneumoniaCNN
from .vit_model import PneumoniaViT

__all__ = ['PneumoniaCNN', 'PneumoniaViT']


