"""Download NIH Chest X-ray dataset in batches."""

import urllib.request
import os
import tarfile
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, filename: str, output_dir: Path):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        filename: Local filename
        output_dir: Directory to save file
    """
    output_path = output_dir / filename
    
    if output_path.exists():
        print(f"File {filename} already exists, skipping download...")
        return output_path
    
    print(f"Downloading {filename}...")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r  Progress: {percent:.1f}%", end='')
    
    try:
        urllib.request.urlretrieve(url, output_path, show_progress)
        print(f"\n  ✓ Downloaded {filename}")
        return output_path
    except Exception as e:
        print(f"\n  ✗ Error downloading {filename}: {e}")
        if output_path.exists():
            output_path.unlink()  # Remove partial download
        return None


def extract_tar_gz(tar_path: Path, extract_dir: Path):
    """
    Extract tar.gz file.
    
    Args:
        tar_path: Path to tar.gz file
        extract_dir: Directory to extract to
    """
    print(f"Extracting {tar_path.name}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print(f"  ✓ Extracted {tar_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error extracting {tar_path.name}: {e}")
        return False


def download_nih_dataset(output_dir: str = "data/nih_dataset", 
                        extract: bool = True,
                        cleanup: bool = False):
    """
    Download NIH Chest X-ray dataset.
    
    Args:
        output_dir: Directory to save downloaded files
        extract: Whether to extract tar.gz files after downloading
        cleanup: Whether to delete tar.gz files after extraction
    """
    # URLs for the zip files (NIH Chest X-ray dataset)
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]
    
    output_dir = Path(output_dir)
    download_dir = output_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("NIH Chest X-ray Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total files to download: {len(links)}\n")
    
    downloaded_files = []
    
    # Download all files
    for idx, link in enumerate(links, 1):
        filename = f'images_{idx:02d}.tar.gz'
        file_path = download_file(link, filename, download_dir)
        if file_path:
            downloaded_files.append(file_path)
    
    print(f"\nDownloaded {len(downloaded_files)}/{len(links)} files")
    
    # Extract if requested
    if extract and downloaded_files:
        print("\n" + "=" * 60)
        print("Extracting files...")
        print("=" * 60)
        
        extract_dir = output_dir / "extracted"
        for tar_file in downloaded_files:
            extract_tar_gz(tar_file, extract_dir)
        
        print(f"\nExtraction complete. Files extracted to: {extract_dir}")
        
        # Cleanup if requested
        if cleanup:
            print("\nCleaning up tar.gz files...")
            for tar_file in downloaded_files:
                tar_file.unlink()
            print("Cleanup complete.")
    
    print("\n" + "=" * 60)
    print("Download process complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Organize images into NORMAL and PNEUMONIA folders")
    print(f"2. Update config/data_config.yaml with dataset path")
    print(f"3. Run: python scripts/prepare_data.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download NIH Chest X-ray dataset')
    parser.add_argument('--output_dir', type=str, default='data/nih_dataset',
                       help='Directory to save downloaded files')
    parser.add_argument('--no-extract', action='store_true',
                       help='Do not extract tar.gz files after downloading')
    parser.add_argument('--cleanup', action='store_true',
                       help='Delete tar.gz files after extraction')
    
    args = parser.parse_args()
    
    download_nih_dataset(
        output_dir=args.output_dir,
        extract=not args.no_extract,
        cleanup=args.cleanup
    )

