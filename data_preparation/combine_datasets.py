"""Combine multiple datasets with stratified splitting."""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import numpy as np
from typing import List, Tuple, Optional


class DatasetCombiner:
    def __init__(self, 
                 dataset1_path,
                 dataset2_path,
                 dataset3_holdout_path,
                 output_dir='data/combined',
                 holdout_dir='data/holdout',
                 train_ratio=0.7,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 random_state=42):
        """
        Combine two datasets for training and keep one as holdout.
        
        Args:
            dataset1_path: Path to first dataset
            dataset2_path: Path to second dataset
            dataset3_holdout_path: Path to holdout dataset
            output_dir: Where to save combined train/val/test splits
            holdout_dir: Where to save holdout dataset
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
        """
        self.dataset1_path = Path(dataset1_path)
        self.dataset2_path = Path(dataset2_path)
        self.dataset3_holdout_path = Path(dataset3_holdout_path)
        self.output_dir = Path(output_dir)
        self.holdout_dir = Path(holdout_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
    
    def collect_images(self, dataset_path):
        """Collect all images from a dataset directory."""
        images = []
        dataset_path = Path(dataset_path)
        
        # Handle different directory structures
        if (dataset_path / 'NORMAL').exists() and (dataset_path / 'PNEUMONIA').exists():
            # Structure: dataset/NORMAL/ and dataset/PNEUMONIA/
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = dataset_path / class_name
                for img_file in class_dir.glob('*.jpeg'):
                    images.append({
                        'path': str(img_file),
                        'class': class_name,
                        'source': dataset_path.name
                    })
                for img_file in class_dir.glob('*.jpg'):
                    images.append({
                        'path': str(img_file),
                        'class': class_name,
                        'source': dataset_path.name
                    })
                for img_file in class_dir.glob('*.png'):
                    images.append({
                        'path': str(img_file),
                        'class': class_name,
                        'source': dataset_path.name
                    })
        else:
            # Alternative structure: dataset/train/NORMAL/, etc.
            for split_dir in ['train', 'val', 'test']:
                split_path = dataset_path / split_dir
                if split_path.exists():
                    for class_name in ['NORMAL', 'PNEUMONIA']:
                        class_dir = split_path / class_name
                        if class_dir.exists():
                            for ext in ['*.jpeg', '*.jpg', '*.png']:
                                for img_file in class_dir.glob(ext):
                                    images.append({
                                        'path': str(img_file),
                                        'class': class_name,
                                        'source': dataset_path.name
                                    })
        
        return images
    
    def combine_datasets(self):
        """Combine dataset1 and dataset2, collect holdout separately."""
        print("Collecting images from datasets...")
        
        # Collect from datasets to combine
        dataset1_images = self.collect_images(self.dataset1_path)
        dataset2_images = self.collect_images(self.dataset2_path)
        
        print(f"Dataset 1: {len(dataset1_images)} images")
        print(f"Dataset 2: {len(dataset2_images)} images")
        
        # Combine the two datasets
        combined_images = dataset1_images + dataset2_images
        
        # Collect holdout dataset
        holdout_images = self.collect_images(self.dataset3_holdout_path)
        print(f"Holdout Dataset: {len(holdout_images)} images")
        
        return combined_images, holdout_images
    
    def create_stratified_splits(self, images):
        """Create stratified train/val/test splits."""
        df = pd.DataFrame(images)
        
        # Print class distribution
        print("\nClass distribution in combined dataset:")
        print(df['class'].value_counts())
        print(f"\nSource distribution:")
        print(df['source'].value_counts())
        
        # Stratified split: first split train from temp (val+test)
        X = df['path'].values
        y = df['class'].values
        
        # First split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=y,
            random_state=self.random_state
        )
        
        # Second split: val vs test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            stratify=y_temp,
            random_state=self.random_state
        )
        
        # Create dataframes for each split
        train_df = df[df['path'].isin(X_train)].copy()
        val_df = df[df['path'].isin(X_val)].copy()
        test_df = df[df['path'].isin(X_test)].copy()
        
        return train_df, val_df, test_df
    
    def copy_files(self, df, target_dir, split_name):
        """Copy files to target directory maintaining class structure."""
        target_path = Path(target_dir) / split_name
        target_path.mkdir(parents=True, exist_ok=True)
        
        for _, row in df.iterrows():
            src_path = Path(row['path'])
            class_name = row['class']
            dst_dir = target_path / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file with original name
            dst_path = dst_dir / src_path.name
            shutil.copy2(src_path, dst_path)
    
    def copy_holdout(self, holdout_images):
        """Copy holdout dataset to holdout directory."""
        print("\nCopying holdout dataset...")
        self.holdout_dir.mkdir(parents=True, exist_ok=True)
        
        for img_info in holdout_images:
            src_path = Path(img_info['path'])
            class_name = img_info['class']
            dst_dir = self.holdout_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            dst_path = dst_dir / src_path.name
            shutil.copy2(src_path, dst_path)
    
    def print_statistics(self, train_df, val_df, test_df, holdout_images):
        """Print detailed statistics about the splits."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        # Combined dataset stats
        print("\nCombined Dataset (Train/Val/Test):")
        print(f"  Total images: {len(train_df) + len(val_df) + len(test_df)}")
        print(f"  Train: {len(train_df)} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        print(f"  Val:   {len(val_df)} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        print(f"  Test:  {len(test_df)} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        
        print("\nTrain set class distribution:")
        print(train_df['class'].value_counts())
        print("\nVal set class distribution:")
        print(val_df['class'].value_counts())
        print("\nTest set class distribution:")
        print(test_df['class'].value_counts())
        
        # Holdout stats
        holdout_df = pd.DataFrame(holdout_images)
        print(f"\nHoldout Dataset:")
        print(f"  Total images: {len(holdout_df)}")
        print("  Class distribution:")
        print(holdout_df['class'].value_counts())
        
        # Source distribution in splits
        print("\nSource distribution in combined splits:")
        print("Train:")
        print(train_df['source'].value_counts())
        print("Val:")
        print(val_df['source'].value_counts())
        print("Test:")
        print(test_df['source'].value_counts())
    
    def process(self):
        """Main processing function."""
        print("Starting dataset combination process...")
        print("="*60)
        
        # Combine datasets
        combined_images, holdout_images = self.combine_datasets()
        
        if len(combined_images) == 0:
            raise ValueError("No images found in datasets to combine!")
        
        # Create stratified splits
        print("\nCreating stratified train/val/test splits...")
        train_df, val_df, test_df = self.create_stratified_splits(combined_images)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to respective directories
        print("\nCopying files to output directories...")
        self.copy_files(train_df, self.output_dir, 'train')
        self.copy_files(val_df, self.output_dir, 'val')
        self.copy_files(test_df, self.output_dir, 'test')
        
        # Copy holdout dataset
        self.copy_holdout(holdout_images)
        
        # Print statistics
        self.print_statistics(train_df, val_df, test_df, holdout_images)
        
        print("\n" + "="*60)
        print("Dataset combination complete!")
        print(f"Combined dataset saved to: {self.output_dir}")
        print(f"Holdout dataset saved to: {self.holdout_dir}")
        print("="*60)
        
        return train_df, val_df, test_df, pd.DataFrame(holdout_images)


