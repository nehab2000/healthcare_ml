"""Duplicate image detection using hashing."""

import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd


def calculate_image_hash(image_path: Path, method: str = 'md5') -> str:
    """
    Calculate hash of image file.
    
    Args:
        image_path: Path to image file
        method: Hash method ('md5' or 'sha256')
        
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.md5() if method == 'md5' else hashlib.sha256()
    
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def calculate_perceptual_hash(image_path: Path) -> str:
    """
    Calculate perceptual hash (pHash) for duplicate detection.
    Uses average hash algorithm - more robust to minor variations.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Hexadecimal hash string
    """
    from PIL import Image
    import numpy as np
    
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize to 8x8
            img = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
            pixels = np.array(img)
            
            # Calculate average
            avg = pixels.mean()
            
            # Create hash: 1 if pixel > avg, 0 otherwise
            hash_bits = (pixels > avg).flatten()
            
            # Convert to hex string
            hash_int = 0
            for bit in hash_bits:
                hash_int = (hash_int << 1) | int(bit)
            
            return hex(hash_int)[2:]
    except Exception as e:
        print(f"Error calculating perceptual hash for {image_path}: {e}")
        return None


def find_duplicates(dataset_paths: List[Path], 
                   method: str = 'md5',
                   use_perceptual: bool = False) -> Dict[str, List[str]]:
    """
    Find duplicate images across datasets.
    
    Args:
        dataset_paths: List of dataset directory paths
        method: Hash method ('md5' or 'sha256')
        use_perceptual: Use perceptual hashing (detects similar images)
        
    Returns:
        Dictionary mapping hash to list of image paths
    """
    hash_to_paths = defaultdict(list)
    
    print(f"Scanning {len(dataset_paths)} datasets for duplicates...")
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        image_files = []
        
        # Find all image files
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(dataset_path.rglob(f'*{ext}'))
        
        print(f"  Processing {len(image_files)} images from {dataset_path.name}...")
        
        for img_file in image_files:
            try:
                if use_perceptual:
                    img_hash = calculate_perceptual_hash(img_file)
                else:
                    img_hash = calculate_image_hash(img_file, method)
                
                if img_hash:
                    hash_to_paths[img_hash].append(str(img_file))
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
    
    # Filter to only duplicates
    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    
    return duplicates


def generate_duplicate_report(duplicates: Dict[str, List[str]]) -> str:
    """
    Generate a report of duplicate images.
    
    Args:
        duplicates: Dictionary of hash to duplicate paths
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("DUPLICATE DETECTION REPORT")
    report.append("=" * 60)
    
    total_duplicates = sum(len(paths) - 1 for paths in duplicates.values())
    unique_duplicate_groups = len(duplicates)
    
    report.append(f"\nTotal duplicate groups: {unique_duplicate_groups}")
    report.append(f"Total duplicate images: {total_duplicates}")
    
    if duplicates:
        report.append("\nDuplicate Groups:")
        for i, (hash_val, paths) in enumerate(list(duplicates.items())[:20], 1):
            report.append(f"\n  Group {i} ({len(paths)} copies):")
            for path in paths:
                report.append(f"    - {path}")
        
        if len(duplicates) > 20:
            report.append(f"\n  ... and {len(duplicates) - 20} more groups")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


def remove_duplicates(duplicates: Dict[str, List[str]], 
                     keep_strategy: str = 'first') -> List[str]:
    """
    Determine which duplicate images to remove.
    
    Args:
        duplicates: Dictionary of hash to duplicate paths
        keep_strategy: 'first' (keep first occurrence) or 'shortest_path' (keep shortest path)
        
    Returns:
        List of paths to remove
    """
    paths_to_remove = []
    
    for hash_val, paths in duplicates.items():
        if keep_strategy == 'first':
            # Keep first, remove rest
            paths_to_remove.extend(paths[1:])
        elif keep_strategy == 'shortest_path':
            # Keep shortest path, remove rest
            paths_sorted = sorted(paths, key=len)
            paths_to_remove.extend(paths_sorted[1:])
    
    return paths_to_remove


