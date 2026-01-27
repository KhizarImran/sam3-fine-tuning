"""
COCO Dataset Validator
Validates downloaded COCO format annotations from Roboflow
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import argparse


def validate_coco_file(json_path):
    """Validate a single COCO JSON file"""
    print(f"\nValidating: {json_path.name}")
    print("-" * 50)

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Failed to read file: {e}")
        return False

    # Check required fields
    required_fields = ['images', 'annotations', 'categories']
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        print(f"  ✗ Missing required fields: {missing_fields}")
        return False

    # Validate structure
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])

    print(f"  ✓ Valid COCO JSON structure")
    print(f"  ✓ Images: {len(images)}")
    print(f"  ✓ Annotations: {len(annotations)}")
    print(f"  ✓ Categories: {len(categories)}")

    # Check categories
    if categories:
        print(f"\n  Categories:")
        for cat in categories:
            cat_id = cat.get('id', 'N/A')
            cat_name = cat.get('name', 'N/A')
            print(f"    - ID {cat_id}: {cat_name}")

    # Annotation statistics
    if annotations:
        # Count annotations per image
        img_annotation_count = defaultdict(int)
        total_area = 0
        min_area = float('inf')
        max_area = 0

        for ann in annotations:
            img_id = ann.get('image_id')
            img_annotation_count[img_id] += 1

            # Check segmentation
            if 'segmentation' in ann:
                seg = ann['segmentation']
                if isinstance(seg, list) and len(seg) > 0:
                    pass  # Valid polygon segmentation
                else:
                    print(f"  ⚠ Warning: Annotation {ann.get('id')} has invalid segmentation")

            # Area statistics
            if 'area' in ann:
                area = ann['area']
                total_area += area
                min_area = min(min_area, area)
                max_area = max(max_area, area)

        avg_annotations = len(annotations) / len(images) if images else 0
        avg_area = total_area / len(annotations) if annotations else 0

        print(f"\n  Annotation Statistics:")
        print(f"    - Avg annotations per image: {avg_annotations:.2f}")
        print(f"    - Avg annotation area: {avg_area:.2f} pixels²")
        print(f"    - Min area: {min_area:.2f} pixels²")
        print(f"    - Max area: {max_area:.2f} pixels²")

        # Check for images without annotations
        images_without_annotations = [img['id'] for img in images if img['id'] not in img_annotation_count]
        if images_without_annotations:
            print(f"  ⚠ Warning: {len(images_without_annotations)} images have no annotations")
            print(f"    Image IDs: {images_without_annotations[:5]}..." if len(images_without_annotations) > 5 else f"    Image IDs: {images_without_annotations}")

    return True


def validate_dataset(dataset_dir):
    """Validate entire dataset directory"""
    print("=" * 70)
    print("COCO DATASET VALIDATOR")
    print("=" * 70)

    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"\n✗ Dataset directory not found: {dataset_dir}")
        print(f"  Did you run roboflow_download.py first?")
        return False

    # Find all subdirectories (Roboflow creates a subdirectory)
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"\n✗ No subdirectories found in {dataset_dir}")
        return False

    # Use the first subdirectory (Roboflow creates one with project name)
    actual_path = subdirs[0]
    print(f"\nDataset location: {actual_path}")

    # Check for train/valid/test splits
    splits = ['train', 'valid', 'test']
    found_splits = []

    for split in splits:
        split_dir = actual_path / split
        if split_dir.exists():
            found_splits.append(split)

    print(f"\nFound splits: {', '.join(found_splits)}")

    # Validate each split
    all_valid = True
    for split in found_splits:
        split_dir = actual_path / split

        # Find COCO JSON file
        json_files = list(split_dir.glob("_annotations.coco.json"))
        if not json_files:
            # Try alternative naming
            json_files = list(split_dir.glob("*.json"))

        if json_files:
            for json_file in json_files:
                valid = validate_coco_file(json_file)
                if not valid:
                    all_valid = False

            # Verify images exist
            image_extensions = ['.jpg', '.jpeg', '.png']
            images = []
            for ext in image_extensions:
                images.extend(split_dir.glob(f"*{ext}"))

            if images:
                print(f"  ✓ Found {len(images)} image files in {split}/")
            else:
                print(f"  ⚠ No image files found in {split}/")
                all_valid = False
        else:
            print(f"\n✗ No COCO JSON files found in {split}/")
            all_valid = False

    # Summary
    print("\n" + "=" * 70)
    if all_valid:
        print("✓ VALIDATION PASSED")
        print("=" * 70)
        print("\nYour dataset is ready for training!")
        print("\nNext steps:")
        print("  1. Visualize annotations: python scripts/visualize_annotations.py")
        print("  2. Configure training: Edit configs/training_config.yaml")
        print("  3. Start training: python scripts/train.py")
    else:
        print("✗ VALIDATION FAILED")
        print("=" * 70)
        print("\nPlease fix the issues above before training.")

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Validate COCO format dataset from Roboflow"
    )
    parser.add_argument(
        "--dataset-dir",
        default="sam3_fuse_cutout_dataset",
        help="Path to dataset directory (default: sam3_fuse_cutout_dataset)"
    )

    args = parser.parse_args()

    validate_dataset(args.dataset_dir)


if __name__ == "__main__":
    main()
