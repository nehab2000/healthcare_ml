"""PyTorch dataset and data loading utilities."""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict
from collections import Counter


class PneumoniaDataset(Dataset):
    """Dataset class for pneumonia X-ray images."""
    
    def __init__(self,
                 data_dir: Path,
                 transform: Optional[transforms.Compose] = None,
                 use_albumentations: bool = True,
                 albumentations_transform: Optional[A.Compose] = None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing NORMAL and PNEUMONIA subdirectories
            transform: PyTorch transforms (used if use_albumentations=False)
            use_albumentations: Whether to use albumentations for augmentation
            albumentations_transform: Albumentations transform pipeline
        """
        self.data_dir = Path(data_dir)
        self.use_albumentations = use_albumentations
        self.transform = transform
        self.albumentations_transform = albumentations_transform
        
        # Collect all images
        self.images = []
        self.labels = []
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    for img_path in class_dir.glob(f'*{ext}'):
                        self.images.append(str(img_path))
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.images)} images from {data_dir}")
        print(f"  NORMAL: {self.labels.count(0)}")
        print(f"  PNEUMONIA: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('L', (224, 224), 0)
        
        # Apply transforms
        if self.use_albumentations and self.albumentations_transform:
            # Convert PIL to numpy for albumentations
            image_np = np.array(image)
            # Albumentations expects 3 channels or grayscale
            if len(image_np.shape) == 2:
                image_np = np.expand_dims(image_np, axis=2)
            augmented = self.albumentations_transform(image=image_np)
            image = augmented['image']
        elif self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = transforms.ToTensor()(image)
        
        return image, label


def get_augmentation_transforms(config: Dict, split: str = 'train') -> A.Compose:
    """
    Get albumentations transform pipeline.
    
    Args:
        config: Configuration dictionary
        split: 'train' or 'val'
        
    Returns:
        Albumentations transform pipeline
    """
    aug_config = config.get('augmentation', {}).get(split, {})
    
    if split == 'train':
        transform_list = [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5 if aug_config.get('horizontal_flip', True) else 0.0),
            A.Rotate(
                limit=aug_config.get('rotation', 15),
                p=0.5 if aug_config.get('rotation', 15) > 0 else 0.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get('brightness', 0.2),
                contrast_limit=aug_config.get('contrast', 0.2),
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats (will be converted to 3 channels)
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    else:
        # Validation/test: minimal augmentation
        transform_list = [
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    
    return A.Compose(transform_list)


def get_class_weights(dataset: PneumoniaDataset) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        dataset: Dataset instance
        
    Returns:
        Tensor of class weights
    """
    labels = dataset.labels
    class_counts = Counter(labels)
    total = len(labels)
    
    # Calculate weights: inverse frequency
    weights = []
    for class_idx in sorted(class_counts.keys()):
        weight = total / (len(class_counts) * class_counts[class_idx])
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def get_data_loaders(data_config: Dict,
                     model_config: Dict,
                     use_weighted_sampler: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_config: Data configuration dictionary
        model_config: Model configuration dictionary
        use_weighted_sampler: Use weighted sampling for class imbalance
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_config['combined_output_dir'])
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    
    # Create datasets
    train_transform = get_augmentation_transforms(model_config, split='train')
    val_transform = get_augmentation_transforms(model_config, split='val')
    test_transform = get_augmentation_transforms(model_config, split='val')  # Same as val
    
    train_dataset = PneumoniaDataset(
        data_dir / 'train',
        use_albumentations=True,
        albumentations_transform=train_transform
    )
    
    val_dataset = PneumoniaDataset(
        data_dir / 'val',
        use_albumentations=True,
        albumentations_transform=val_transform
    )
    
    test_dataset = PneumoniaDataset(
        data_dir / 'test',
        use_albumentations=True,
        albumentations_transform=test_transform
    )
    
    # Create samplers
    train_sampler = None
    if use_weighted_sampler:
        class_weights = get_class_weights(train_dataset)
        sample_weights = [class_weights[label] for label in train_dataset.labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_holdout_loader(data_config: Dict,
                      model_config: Dict) -> DataLoader:
    """
    Create data loader for holdout set.
    
    Args:
        data_config: Data configuration dictionary
        model_config: Model configuration dictionary
        
    Returns:
        DataLoader for holdout set
    """
    holdout_dir = Path(data_config['holdout_output_dir'])
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    
    transform = get_augmentation_transforms(model_config, split='val')
    
    holdout_dataset = PneumoniaDataset(
        holdout_dir,
        use_albumentations=True,
        albumentations_transform=transform
    )
    
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return holdout_loader


