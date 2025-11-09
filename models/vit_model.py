"""Vision Transformer model for pneumonia detection."""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from typing import Optional


class PneumoniaViT(nn.Module):
    """Vision Transformer model for pneumonia detection."""
    
    def __init__(self,
                 model_name: str = 'google/vit-base-patch16-224',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 image_size: int = 224,
                 patch_size: int = 16,
                 dropout_rate: float = 0.5):
        """
        Initialize Vision Transformer model.
        
        Args:
            model_name: HuggingFace model name or path
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            image_size: Input image size
            patch_size: Patch size for ViT
            dropout_rate: Dropout rate for classification head
        """
        super(PneumoniaViT, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        
        if pretrained:
            try:
                self.vit = ViTModel.from_pretrained(model_name)
                hidden_size = self.vit.config.hidden_size
            except Exception as e:
                print(f"Warning: Could not load pretrained model {model_name}: {e}")
                print("Initializing from scratch...")
                config = ViTConfig(
                    image_size=image_size,
                    patch_size=patch_size,
                    num_channels=3,  # RGB
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    hidden_size=768,
                    intermediate_size=3072
                )
                self.vit = ViTModel(config)
                hidden_size = 768
        else:
            config = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                num_channels=3,  # RGB
                num_hidden_layers=12,
                num_attention_heads=12,
                hidden_size=768,
                intermediate_size=3072
            )
            self.vit = ViTModel(config)
            hidden_size = 768
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
               If grayscale (1 channel), will be converted to RGB
        """
        # ViT expects 3 channels, convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # ViT expects pixel_values
        outputs = self.vit(pixel_values=x)
        pooled_output = outputs.pooler_output
        
        return self.classifier(pooled_output)
    
    def get_attention_weights(self, x):
        """
        Get attention weights for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights from all layers
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        outputs = self.vit(pixel_values=x, output_attentions=True)
        return outputs.attentions


def create_vit_model(config: dict) -> PneumoniaViT:
    """
    Create ViT model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized ViT model
    """
    vit_config = config.get('vit', {})
    return PneumoniaViT(
        model_name=vit_config.get('model_name', 'google/vit-base-patch16-224'),
        num_classes=vit_config.get('num_classes', 2),
        pretrained=vit_config.get('pretrained', True),
        image_size=vit_config.get('image_size', 224),
        patch_size=vit_config.get('patch_size', 16),
        dropout_rate=vit_config.get('dropout_rate', 0.5)
    )


