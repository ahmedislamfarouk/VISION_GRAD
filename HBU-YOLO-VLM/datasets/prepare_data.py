"""
Dataset Preparation Scripts for HBU-YOLO-VLM
"""

import os
import json
import shutil
from pathlib import Path
import argparse
import zipfile
import requests
from tqdm import tqdm


def download_file(url: str, destination: str):
    """Download file from URL"""
    print(f"Downloading from {url}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def prepare_xbd_dataset(output_dir: str = "datasets/xBD"):
    """
    Prepare xBD dataset
    
    Download from: https://xview2.org/dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("xBD Dataset Preparation")
    print("="*50)
    
    # xBD dataset needs to be downloaded manually from xView2 website
    # Instructions:
    print("\nPlease download xBD dataset from: https://xview2.org/dataset")
    print("\nAfter downloading, extract to:")
    print(f"  {output_dir}/")
    print("\nExpected structure:")
    print(f"  {output_dir}/train/images/")
    print(f"  {output_dir}/train/labels/")
    print(f"  {output_dir}/val/images/")
    print(f"  {output_dir}/val/labels/")
    print(f"  {output_dir}/test/images/")
    print(f"  {output_dir}/test/labels/")
    
    # Check if dataset exists
    if (output_dir / "train" / "images").exists():
        print("\n✓ xBD dataset found!")
    else:
        print("\n✗ xBD dataset not found. Please download and extract.")


def prepare_rescuenet_dataset(output_dir: str = "datasets/RescueNet"):
    """
    Prepare RescueNet dataset
    
    GitHub: https://github.com/remi-md/rescuenet
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("RescueNet Dataset Preparation")
    print("="*50)
    
    # RescueNet can be downloaded from GitHub releases
    print("\nPlease download RescueNet dataset from:")
    print("  https://github.com/remi-md/rescuenet")
    print("\nAfter downloading, extract to:")
    print(f"  {output_dir}/")
    
    # Check if dataset exists
    if (output_dir / "train" / "images").exists():
        print("\n✓ RescueNet dataset found!")
    else:
        print("\n✗ RescueNet dataset not found. Please download.")


def prepare_floodnet_dataset(output_dir: str = "datasets/FloodNet"):
    """
    Prepare FloodNet dataset
    
    GitHub: https://github.com/Ankush191/FloodNet
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("FloodNet Dataset Preparation")
    print("="*50)
    
    print("\nPlease download FloodNet dataset from:")
    print("  https://github.com/Ankush191/FloodNet")
    print("\nAfter downloading, extract to:")
    print(f"  {output_dir}/")
    
    # Check if dataset exists
    if (output_dir / "train" / "images").exists():
        print("\n✓ FloodNet dataset found!")
    else:
        print("\n✗ FloodNet dataset not found. Please download.")


def prepare_combined_dataset(
    xbd_dir: str = "datasets/xBD",
    rescuenet_dir: str = "datasets/RescueNet",
    floodnet_dir: str = "datasets/FloodNet",
    output_dir: str = "datasets/combined"
):
    """
    Create combined dataset from all sources
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("Combined Dataset Preparation")
    print("="*50)
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Copy images and labels from each dataset
    datasets = [
        ("xbd", Path(xbd_dir)),
        ("rescuenet", Path(rescuenet_dir)),
        ("floodnet", Path(floodnet_dir))
    ]
    
    for dataset_name, dataset_dir in datasets:
        if not dataset_dir.exists():
            print(f"\n⚠ {dataset_name} not found at {dataset_dir}")
            continue
        
        print(f"\nProcessing {dataset_name}...")
        
        for split in ["train", "val", "test"]:
            split_dir = dataset_dir / split
            
            if not split_dir.exists():
                continue
            
            # Copy images
            images_dir = split_dir / "images"
            if images_dir.exists():
                for img_path in images_dir.glob("*.jpg"):
                    new_name = f"{dataset_name}_{img_path.name}"
                    shutil.copy(img_path, output_dir / split / "images" / new_name)
            
            # Copy labels
            labels_dir = split_dir / "labels"
            if labels_dir.exists():
                for label_path in labels_dir.glob("*.json"):
                    new_name = f"{dataset_name}_{label_path.name}"
                    shutil.copy(label_path, output_dir / split / "labels" / new_name)
    
    print("\n✓ Combined dataset created!")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser("Dataset Preparation")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["xbd", "rescuenet", "floodnet", "combined", "all"],
        help="Dataset to prepare"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Base output directory"
    )
    
    args = parser.parse_args()
    
    output_base = Path(args.output_dir)
    
    if args.dataset in ["all", "xbd"]:
        prepare_xbd_dataset(output_base / "xBD")
    
    if args.dataset in ["all", "rescuenet"]:
        prepare_rescuenet_dataset(output_base / "RescueNet")
    
    if args.dataset in ["all", "floodnet"]:
        prepare_floodnet_dataset(output_base / "FloodNet")
    
    if args.dataset in ["all", "combined"]:
        prepare_combined_dataset(
            output_base / "xBD",
            output_base / "RescueNet",
            output_base / "FloodNet",
            output_base / "combined"
        )


if __name__ == "__main__":
    main()
