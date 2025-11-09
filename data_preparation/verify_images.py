"""Image verification and cleaning utilities."""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd


class ImageVerifier:
    """Verify and clean images to ensure consistency across datasets."""
    
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 allowed_formats: List[str] = None,
                 require_grayscale: bool = True,
                 min_file_size: int = 1000,  # bytes
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 auto_resize: bool = True,
                 auto_convert_format: bool = True,
                 remove_invalid: bool = False):
        """
        Initialize image verifier.
        
        Args:
            target_size: Target (width, height) for all images
            allowed_formats: List of allowed formats (e.g., ['JPEG', 'PNG'])
            require_grayscale: Whether to require grayscale images
            min_file_size: Minimum file size in bytes
            max_file_size: Maximum file size in bytes
            auto_resize: Automatically resize images to target size
            auto_convert_format: Automatically convert formats if needed
            remove_invalid: Remove invalid images instead of reporting only
        """
        self.target_size = target_size
        self.allowed_formats = allowed_formats or ['JPEG', 'PNG', 'JPG']
        self.require_grayscale = require_grayscale
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.auto_resize = auto_resize
        self.auto_convert_format = auto_convert_format
        self.remove_invalid = remove_invalid
        
        self.issues = defaultdict(list)
        self.stats = defaultdict(int)
    
    def verify_image(self, image_path: Path) -> Dict:
        """
        Verify a single image.
        
        Returns:
            Dictionary with verification results
        """
        result = {
            'path': str(image_path),
            'valid': True,
            'issues': [],
            'size': None,
            'format': None,
            'mode': None,
            'file_size': None
        }
        
        try:
            # Check file size
            file_size = image_path.stat().st_size
            result['file_size'] = file_size
            
            if file_size < self.min_file_size:
                result['valid'] = False
                result['issues'].append(f'File too small: {file_size} bytes')
                self.issues['too_small'].append(str(image_path))
            
            if file_size > self.max_file_size:
                result['valid'] = False
                result['issues'].append(f'File too large: {file_size} bytes')
                self.issues['too_large'].append(str(image_path))
            
            # Try to open and verify image
            try:
                with Image.open(image_path) as img:
                    result['size'] = img.size  # (width, height)
                    result['format'] = img.format
                    result['mode'] = img.mode
                    
                    # Check format
                    if img.format not in self.allowed_formats:
                        result['valid'] = False
                        result['issues'].append(f'Invalid format: {img.format}')
                        self.issues['invalid_format'].append(str(image_path))
                    
                    # Check size
                    if img.size != self.target_size:
                        result['valid'] = False
                        result['issues'].append(f'Wrong size: {img.size}, expected {self.target_size}')
                        self.issues['wrong_size'].append(str(image_path))
                    
                    # Check color mode
                    if self.require_grayscale:
                        if img.mode not in ['L', 'LA']:  # L = grayscale, LA = grayscale with alpha
                            result['valid'] = False
                            result['issues'].append(f'Not grayscale: {img.mode}')
                            self.issues['wrong_mode'].append(str(image_path))
                    
                    # Check if image is corrupted
                    try:
                        img.verify()
                    except Exception as e:
                        result['valid'] = False
                        result['issues'].append(f'Corrupted image: {str(e)}')
                        self.issues['corrupted'].append(str(image_path))
                    
                    # Check if image is blank (all same pixel value)
                    img_array = np.array(img.convert('L'))
                    if np.all(img_array == img_array[0, 0]):
                        result['valid'] = False
                        result['issues'].append('Blank image (all pixels same value)')
                        self.issues['blank'].append(str(image_path))
                    
            except Exception as e:
                result['valid'] = False
                result['issues'].append(f'Cannot open image: {str(e)}')
                self.issues['cannot_open'].append(str(image_path))
        
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f'File error: {str(e)}')
            self.issues['file_error'].append(str(image_path))
        
        if result['valid']:
            self.stats['valid'] += 1
        else:
            self.stats['invalid'] += 1
        
        return result
    
    def verify_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """
        Verify all images in a dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            DataFrame with verification results
        """
        results = []
        dataset_path = Path(dataset_path)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.rglob(f'*{ext}'))
        
        print(f"Found {len(image_files)} images in {dataset_path}")
        
        for img_path in image_files:
            result = self.verify_image(img_path)
            results.append(result)
        
        df = pd.DataFrame(results)
        return df
    
    def clean_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Clean and standardize a single image.
        
        Args:
            image_path: Path to source image
            output_path: Path to save cleaned image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale if required
                if self.require_grayscale and img.mode not in ['L', 'LA']:
                    img = img.convert('L')
                
                # Resize to target size
                if img.size != self.target_size:
                    img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                
                # Convert format if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save in JPEG format (standard for medical imaging)
                if img.mode == 'LA':
                    img = img.convert('L')
                
                img.save(output_path, 'JPEG', quality=95)
                return True
        
        except Exception as e:
            print(f"Error cleaning {image_path}: {e}")
            return False
    
    def clean_dataset(self, dataset_path: Path, output_path: Path) -> Dict:
        """
        Clean entire dataset and save to output path.
        
        Args:
            dataset_path: Path to source dataset
            output_path: Path to save cleaned dataset
            
        Returns:
            Dictionary with cleaning statistics
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.rglob(f'*{ext}'))
        
        cleaned_count = 0
        failed_count = 0
        
        for img_path in image_files:
            # Preserve directory structure
            relative_path = img_path.relative_to(dataset_path)
            output_file = output_path / relative_path
            output_file = output_file.with_suffix('.jpg')  # Convert to JPEG
            
            if self.clean_image(img_path, output_file):
                cleaned_count += 1
            else:
                failed_count += 1
        
        return {
            'total': len(image_files),
            'cleaned': cleaned_count,
            'failed': failed_count
        }
    
    def generate_report(self, verification_df: pd.DataFrame) -> str:
        """
        Generate a detailed verification report.
        
        Args:
            verification_df: DataFrame with verification results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("IMAGE VERIFICATION REPORT")
        report.append("=" * 60)
        
        total = len(verification_df)
        valid = verification_df['valid'].sum()
        invalid = total - valid
        
        report.append(f"\nTotal images: {total}")
        report.append(f"Valid images: {valid} ({valid/total*100:.1f}%)")
        report.append(f"Invalid images: {invalid} ({invalid/total*100:.1f}%)")
        
        # Issue breakdown
        report.append("\nIssue Breakdown:")
        for issue_type, paths in self.issues.items():
            report.append(f"  {issue_type}: {len(paths)} images")
        
        # Size distribution
        if 'size' in verification_df.columns:
            sizes = verification_df['size'].dropna()
            if len(sizes) > 0:
                report.append("\nSize Distribution:")
                size_counts = sizes.value_counts()
                for size, count in size_counts.head(10).items():
                    report.append(f"  {size}: {count} images")
        
        # Format distribution
        if 'format' in verification_df.columns:
            formats = verification_df['format'].dropna()
            if len(formats) > 0:
                report.append("\nFormat Distribution:")
                format_counts = formats.value_counts()
                for fmt, count in format_counts.items():
                    report.append(f"  {fmt}: {count} images")
        
        # Mode distribution
        if 'mode' in verification_df.columns:
            modes = verification_df['mode'].dropna()
            if len(modes) > 0:
                report.append("\nColor Mode Distribution:")
                mode_counts = modes.value_counts()
                for mode, count in mode_counts.items():
                    report.append(f"  {mode}: {count} images")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def verify_and_clean(self, dataset_path: Path, output_path: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
        """
        Verify dataset and optionally clean it.
        
        Args:
            dataset_path: Path to dataset
            output_path: Optional path to save cleaned dataset
            
        Returns:
            Tuple of (verification DataFrame, report string)
        """
        # Verify
        verification_df = self.verify_dataset(dataset_path)
        report = self.generate_report(verification_df)
        
        # Clean if requested
        if output_path:
            print("\nCleaning dataset...")
            clean_stats = self.clean_dataset(dataset_path, output_path)
            report += f"\nCleaning Statistics:\n"
            report += f"  Total: {clean_stats['total']}\n"
            report += f"  Cleaned: {clean_stats['cleaned']}\n"
            report += f"  Failed: {clean_stats['failed']}\n"
        
        return verification_df, report


