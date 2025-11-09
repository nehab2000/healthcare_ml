"""CNN model architectures for pneumonia detection."""

import torch
import torch.nn as nn
from torchvision import models
import timm
from typing import Optional


class PneumoniaCNN(nn.Module):
    """CNN model for pneumonia detection using various backbones."""
    
    def __init__(self, 
                 architecture: str = 'resnet50',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):
        """
        Initialize CNN model.
        
        Args:
            architecture: Backbone architecture ('resnet50', 'resnet101', 'densenet121', 'efficientnet_b3')
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            dropout_rate: Dropout rate for classification head
        """
        super(PneumoniaCNN, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        
        if architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = self._create_classifier(num_features, dropout_rate)
        
        elif architecture == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = self._create_classifier(num_features, dropout_rate)
        
        elif architecture == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = self._create_classifier(num_features, dropout_rate)
        
        elif architecture.startswith('efficientnet'):
            # Use timm for EfficientNet
            self.backbone = timm.create_model(
                architecture,
                pretrained=pretrained,
                num_classes=0  # Remove classifier
            )
            num_features = self.backbone.num_features
            self.classifier = self._create_classifier(num_features, dropout_rate)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def _create_classifier(self, num_features: int, dropout_rate: float) -> nn.Module:
        """Create classification head."""
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.6),  # Slightly less dropout
            nn.Linear(512, self.num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        if self.architecture.startswith('efficientnet'):
            features = self.backbone(x)
            return self.classifier(features)
        else:
            return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before classification head (for visualization)."""
        if self.architecture == 'resnet50' or self.architecture == 'resnet101':
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        elif self.architecture == 'densenet121':
            features = self.backbone.features(x)
            out = nn.functional.relu(features, inplace=True)
            out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            return out
        elif self.architecture.startswith('efficientnet'):
            return self.backbone(x)
        else:
            raise ValueError(f"Feature extraction not implemented for {self.architecture}")


def create_cnn_model(config: dict) -> PneumoniaCNN:
    """
    Create CNN model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized CNN model
    """
    cnn_config = config.get('cnn', {})
    return PneumoniaCNN(
        architecture=cnn_config.get('architecture', 'resnet50'),
        num_classes=cnn_config.get('num_classes', 2),
        pretrained=cnn_config.get('pretrained', True),
        dropout_rate=cnn_config.get('dropout_rate', 0.5)
    )


