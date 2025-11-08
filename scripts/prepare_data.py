"""
Main script for data preparation: verification and combination.

This is the FIRST script you should run after organizing your datasets.
It will:
1. Verify all images (size, format, quality)
2. Detect duplicate images
3. Combine dataset1 and dataset2
4. Create stratified train/val/test splits
5. Copy holdout dataset separately

Usage:
    python scripts/prepare_data.py

Make sure to update config/data_config.yaml with your dataset paths first!
"""

import sys
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_preparation.verify_images import ImageVerifier
from data_preparation.combine_datasets import DatasetCombiner
from data_preparation.detect_duplicates import find_duplicates, generate_duplicate_report


def main():
    """Main data preparation workflow."""
    # Load configuration
    config_path = Path('config/data_config.yaml')
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        print("Please create config/data_config.yaml")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    verification_config = data_config.get('verification', {})
    
    # Get paths
    dataset1_path = Path(data_config['dataset1_path'])
    dataset2_path = Path(data_config['dataset2_path'])
    dataset3_holdout_path = Path(data_config['dataset3_holdout_path'])
    
    # Verify paths exist
    for name, path in [('Dataset 1', dataset1_path), 
                       ('Dataset 2', dataset2_path),
                       ('Holdout Dataset', dataset3_holdout_path)]:
        if not path.exists():
            print(f"Error: {name} not found at {path}")
            return
    
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Verify images
    print("\n[Step 1/4] Verifying images...")
    target_size = tuple(verification_config.get('target_size', [224, 224]))
    
    verifier = ImageVerifier(
        target_size=target_size,
        allowed_formats=verification_config.get('allowed_formats', ['JPEG', 'PNG']),
        require_grayscale=verification_config.get('require_grayscale', True),
        min_file_size=verification_config.get('min_file_size', 1000),
        max_file_size=verification_config.get('max_file_size', 10 * 1024 * 1024),
        auto_resize=verification_config.get('auto_resize', True),
        auto_convert_format=verification_config.get('auto_convert_format', True),
        remove_invalid=verification_config.get('remove_invalid', False)
    )
    
    # Verify all three datasets
    print(f"\nVerifying {dataset1_path.name}...")
    df1, report1 = verifier.verify_and_clean(dataset1_path)
    print(report1)
    
    print(f"\nVerifying {dataset2_path.name}...")
    verifier2 = ImageVerifier(
        target_size=target_size,
        allowed_formats=verification_config.get('allowed_formats', ['JPEG', 'PNG']),
        require_grayscale=verification_config.get('require_grayscale', True),
        min_file_size=verification_config.get('min_file_size', 1000),
        max_file_size=verification_config.get('max_file_size', 10 * 1024 * 1024),
        auto_resize=verification_config.get('auto_resize', True),
        auto_convert_format=verification_config.get('auto_convert_format', True),
        remove_invalid=verification_config.get('remove_invalid', False)
    )
    df2, report2 = verifier2.verify_and_clean(dataset2_path)
    print(report2)
    
    print(f"\nVerifying {dataset3_holdout_path.name}...")
    verifier3 = ImageVerifier(
        target_size=target_size,
        allowed_formats=verification_config.get('allowed_formats', ['JPEG', 'PNG']),
        require_grayscale=verification_config.get('require_grayscale', True),
        min_file_size=verification_config.get('min_file_size', 1000),
        max_file_size=verification_config.get('max_file_size', 10 * 1024 * 1024),
        auto_resize=verification_config.get('auto_resize', True),
        auto_convert_format=verification_config.get('auto_convert_format', True),
        remove_invalid=verification_config.get('remove_invalid', False)
    )
    df3, report3 = verifier3.verify_and_clean(dataset3_holdout_path)
    print(report3)
    
    # Check if verification passed
    total_invalid = (df1['valid'] == False).sum() + (df2['valid'] == False).sum() + (df3['valid'] == False).sum()
    if total_invalid > 0:
        print(f"\nWarning: {total_invalid} invalid images found across datasets.")
        if not verification_config.get('remove_invalid', False):
            print("Consider enabling 'remove_invalid' in config or manually fixing issues.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    
    # Step 2: Detect duplicates
    print("\n[Step 2/4] Detecting duplicates...")
    duplicates = find_duplicates(
        [dataset1_path, dataset2_path, dataset3_holdout_path],
        method='md5',
        use_perceptual=verification_config.get('use_perceptual_hashing', False)
    )
    
    if duplicates:
        duplicate_report = generate_duplicate_report(duplicates)
        print(duplicate_report)
        print(f"\nWarning: {len(duplicates)} duplicate groups found.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print("No duplicates found.")
    
    # Step 3: Combine datasets
    print("\n[Step 3/4] Combining datasets...")
    combiner = DatasetCombiner(
        dataset1_path=dataset1_path,
        dataset2_path=dataset2_path,
        dataset3_holdout_path=dataset3_holdout_path,
        output_dir=data_config['combined_output_dir'],
        holdout_dir=data_config['holdout_output_dir'],
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        random_state=data_config['random_seed']
    )
    
    train_df, val_df, test_df, holdout_df = combiner.process()
    
    # Step 4: Final verification of combined dataset
    print("\n[Step 4/4] Final verification of combined dataset...")
    combined_path = Path(data_config['combined_output_dir'])
    final_verifier = ImageVerifier(
        target_size=target_size,
        allowed_formats=verification_config.get('allowed_formats', ['JPEG', 'PNG']),
        require_grayscale=verification_config.get('require_grayscale', True)
    )
    
    for split in ['train', 'val', 'test']:
        split_path = combined_path / split
        if split_path.exists():
            print(f"\nVerifying {split} set...")
            df_split, report_split = final_verifier.verify_and_clean(split_path)
            invalid_count = (df_split['valid'] == False).sum()
            if invalid_count > 0:
                print(f"Warning: {invalid_count} invalid images in {split} set")
            else:
                print(f"✓ {split} set verified successfully")
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print("1. Review the statistics above")
    print("2. Train models using: python training/train_cnn.py or python training/train_vit.py")
    print("3. Evaluate on test set and holdout set")


if __name__ == "__main__":
    main()


