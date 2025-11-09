"""Download Kaggle Chest X-ray Pneumonia dataset."""

import kagglehub
from pathlib import Path
import shutil
from tqdm import tqdm
import os


def download_kaggle_dataset(output_dir: str = "data/kaggle_dataset",
                            organize: bool = True):
    """
    Download Kaggle Chest X-ray Pneumonia dataset.
    
    Args:
        output_dir: Directory to save dataset
        organize: Whether to organize images into NORMAL/PNEUMONIA folders
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Kaggle Chest X-ray Pneumonia Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}\n")
    
    # Download dataset
    print("Downloading dataset from Kaggle...")
    print("This may take a few minutes depending on your internet connection...\n")
    
    try:
        dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"✓ Dataset downloaded successfully!")
        print(f"  Path: {dataset_path}\n")
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure kagglehub is installed: pip install kagglehub")
        print("2. You may need to authenticate with Kaggle API")
        print("   - Get API credentials from: https://www.kaggle.com/settings")
        print("   - Place kaggle.json in ~/.kaggle/ (or set KAGGLE_CONFIG_DIR)")
        return False
    
    # Check dataset structure
    dataset_path = Path(dataset_path)
    print(f"Exploring dataset structure at: {dataset_path}\n")
    
    # Find the main data directory
    # Kaggle datasets often have the structure: dataset_name/chest_xray/train/test/val
    possible_paths = [
        dataset_path / "chest_xray",
        dataset_path / "ChestXRay2017",
        dataset_path / "chest-xray-pneumonia",
        dataset_path
    ]
    
    data_root = None
    for path in possible_paths:
        if path.exists():
            # Check if it has train/test/val or NORMAL/PNEUMONIA structure
            if (path / "train").exists() or (path / "NORMAL").exists():
                data_root = path
                break
    
    if data_root is None:
        # List contents to help debug
        print("Could not find expected dataset structure. Contents:")
        for item in dataset_path.iterdir():
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        print("\nPlease manually organize the dataset or check the structure.")
        return False
    
    print(f"Found dataset root: {data_root}\n")
    
    # Organize dataset if requested
    if organize:
        print("Organizing dataset into NORMAL/PNEUMONIA structure...")
        organized_path = output_dir / "organized"
        organized_path.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already has train/test/val structure
        if (data_root / "train").exists():
            # Dataset has train/test/val structure
            print("Dataset has train/test/val structure. Organizing...")
            
            for split in ["train", "test", "val"]:
                split_dir = data_root / split
                if split_dir.exists():
                    print(f"\nProcessing {split} set...")
                    
                    for class_name in ["NORMAL", "PNEUMONIA"]:
                        source_dir = split_dir / class_name
                        if source_dir.exists():
                            # Count images
                            image_files = list(source_dir.glob("*.jpeg")) + \
                                        list(source_dir.glob("*.jpg")) + \
                                        list(source_dir.glob("*.png"))
                            
                            if image_files:
                                # Create organized structure
                                target_dir = organized_path / split / class_name
                                target_dir.mkdir(parents=True, exist_ok=True)
                                
                                # Copy images
                                print(f"  Copying {len(image_files)} {class_name} images...")
                                for img_file in tqdm(image_files, desc=f"    {class_name}", leave=False):
                                    shutil.copy2(img_file, target_dir / img_file.name)
                                
                                print(f"  ✓ Copied {len(image_files)} {class_name} images to {target_dir}")
        
        elif (data_root / "NORMAL").exists():
            # Dataset already has NORMAL/PNEUMONIA structure
            print("Dataset already has NORMAL/PNEUMONIA structure.")
            print("Copying to organized directory...")
            
            for class_name in ["NORMAL", "PNEUMONIA"]:
                source_dir = data_root / class_name
                if source_dir.exists():
                    image_files = list(source_dir.glob("*.jpeg")) + \
                                list(source_dir.glob("*.jpg")) + \
                                list(source_dir.glob("*.png"))
                    
                    if image_files:
                        target_dir = organized_path / class_name
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        print(f"  Copying {len(image_files)} {class_name} images...")
                        for img_file in tqdm(image_files, desc=f"    {class_name}", leave=False):
                            shutil.copy2(img_file, target_dir / img_file.name)
                        
                        print(f"  ✓ Copied {len(image_files)} {class_name} images")
        
        else:
            print("Warning: Could not determine dataset structure.")
            print("Please manually organize the dataset.")
            print(f"Dataset is located at: {data_root}")
            return False
        
        # Print statistics
        print("\n" + "=" * 60)
        print("Dataset Organization Complete")
        print("=" * 60)
        
        organized_path = organized_path
        if (organized_path / "train").exists():
            for split in ["train", "test", "val"]:
                split_path = organized_path / split
                if split_path.exists():
                    normal_count = len(list((split_path / "NORMAL").glob("*.jpeg"))) + \
                                 len(list((split_path / "NORMAL").glob("*.jpg"))) + \
                                 len(list((split_path / "NORMAL").glob("*.png")))
                    pneu_count = len(list((split_path / "PNEUMONIA").glob("*.jpeg"))) + \
                               len(list((split_path / "PNEUMONIA").glob("*.jpg"))) + \
                               len(list((split_path / "PNEUMONIA").glob("*.png")))
                    
                    print(f"\n{split.upper()} set:")
                    print(f"  NORMAL:    {normal_count} images")
                    print(f"  PNEUMONIA: {pneu_count} images")
                    print(f"  Total:     {normal_count + pneu_count} images")
        else:
            normal_count = len(list((organized_path / "NORMAL").glob("*.jpeg"))) + \
                         len(list((organized_path / "NORMAL").glob("*.jpg"))) + \
                         len(list((organized_path / "NORMAL").glob("*.png")))
            pneu_count = len(list((organized_path / "PNEUMONIA").glob("*.jpeg"))) + \
                       len(list((organized_path / "PNEUMONIA").glob("*.jpg"))) + \
                       len(list((organized_path / "PNEUMONIA").glob("*.png")))
            
            print(f"\nTotal images:")
            print(f"  NORMAL:    {normal_count} images")
            print(f"  PNEUMONIA: {pneu_count} images")
            print(f"  Total:     {normal_count + pneu_count} images")
        
        print(f"\nOrganized dataset saved to: {organized_path}")
        print("\nNext steps:")
        print("1. If dataset has train/test/val splits, you can use them directly")
        print("2. Or combine with other datasets using: python scripts/prepare_data.py")
        print("3. Update config/data_config.yaml with the path to this dataset")
    
    else:
        print(f"\nDataset downloaded to: {dataset_path}")
        print("Skipping organization. You can organize manually if needed.")
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download Kaggle Chest X-ray Pneumonia dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/kaggle_dataset',
        help='Directory to save downloaded dataset'
    )
    parser.add_argument(
        '--no-organize',
        action='store_true',
        help='Do not organize images into NORMAL/PNEUMONIA folders'
    )
    
    args = parser.parse_args()
    
    download_kaggle_dataset(
        output_dir=args.output_dir,
        organize=not args.no_organize
    )


if __name__ == "__main__":
    main()

