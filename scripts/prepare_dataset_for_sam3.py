"""
Prepare Roboflow COCO dataset for SAM3 training

Converts the downloaded Roboflow dataset into SAM3's expected format:
- Organizes into train/valid/test splits
- Ensures COCO annotations are compatible
- Creates dataset metadata
"""

import json
import os
import shutil
import argparse
from pathlib import Path


def prepare_sam3_dataset(source_dir, output_dir, dataset_name="fuse-cutout-detection"):
    """
    Prepare dataset for SAM3 training

    Args:
        source_dir: Path to downloaded Roboflow dataset (e.g., fuse-netrual-training-dataset/)
        output_dir: Path to output SAM3-formatted dataset
        dataset_name: Name for the dataset (used in directory structure)
    """
    print("=" * 70)
    print("SAM3 DATASET PREPARATION")
    print("=" * 70)

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Check if source_dir contains train/valid/test directly
    # or if there's a subdirectory that contains them (Roboflow sometimes does this)
    has_splits_directly = (source_path / "train").exists() or \
                          (source_path / "valid").exists() or \
                          (source_path / "test").exists()

    if has_splits_directly:
        actual_source = source_path
    else:
        # Look for subdirectory containing train/valid/test
        subdirs = [d for d in source_path.iterdir() if d.is_dir()]
        actual_source = subdirs[0] if subdirs else source_path

    print(f"\nSource: {actual_source}")
    print(f"Output: {output_path}")

    # Create output directory structure
    # SAM3 expects: dataset_name/train, dataset_name/valid, dataset_name/test
    dataset_output = output_path / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    print(f"\n1. Creating directory structure...")

    # Process each split (train, valid, test)
    splits = ['train', 'valid', 'test']
    total_images = 0
    total_annotations = 0
    last_coco_data = None  # Keep track of last valid COCO data for metadata

    for split in splits:
        split_source = actual_source / split

        if not split_source.exists():
            print(f"   ⚠ {split}/ not found, skipping")
            continue

        split_output = dataset_output / split
        split_output.mkdir(exist_ok=True)

        # Find COCO JSON file
        json_files = list(split_source.glob("_annotations.coco.json"))
        if not json_files:
            json_files = list(split_source.glob("*.json"))

        if not json_files:
            print(f"   ⚠ No COCO JSON found in {split}/")
            continue

        json_file = json_files[0]

        # Load and process COCO annotations
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        last_coco_data = coco_data  # Save for metadata creation later

        num_images = len(coco_data.get('images', []))
        num_annotations = len(coco_data.get('annotations', []))
        total_images += num_images
        total_annotations += num_annotations

        print(f"   ✓ {split}/: {num_images} images, {num_annotations} annotations")

        # Copy images
        for img_info in coco_data['images']:
            img_filename = img_info['file_name']
            src_img = split_source / img_filename
            dst_img = split_output / img_filename

            if src_img.exists():
                shutil.copy2(src_img, dst_img)

        # Save COCO annotations
        output_json = split_output / "_annotations.coco.json"
        with open(output_json, 'w') as f:
            json.dump(coco_data, f, indent=2)

    print(f"\n2. Dataset summary:")
    print(f"   Total images: {total_images}")
    print(f"   Total annotations: {total_annotations}")

    if total_images == 0:
        print("\n✗ ERROR: No images found!")
        print(f"   Checked for splits in: {actual_source}")
        print(f"   Make sure the dataset has train/, valid/, or test/ directories")
        return None

    # Create dataset metadata file
    metadata = {
        "dataset_name": dataset_name,
        "total_images": total_images,
        "total_annotations": total_annotations,
        "categories": last_coco_data.get('categories', []) if last_coco_data else [],
        "format": "COCO",
        "splits": ["train", "valid", "test"]
    }

    metadata_file = dataset_output / "dataset_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n3. Created metadata file: {metadata_file.name}")

    # Create a text file with category mapping (SAM3 needs text prompts)
    if last_coco_data:
        categories_file = dataset_output / "categories.txt"
        with open(categories_file, 'w') as f:
            for cat in last_coco_data.get('categories', []):
                # Write category name (used as text prompt in SAM3)
                f.write(f"{cat['name']}\n")

        print(f"   Created categories file: {categories_file.name}")

    # Check for potential issues
    print(f"\n4. Validation checks:")

    warnings = []

    if total_images < 100:
        warnings.append(f"Only {total_images} images (recommended: 100+ for pilot)")

    # Check if only train split exists
    has_valid = (dataset_output / "valid").exists() and len(list((dataset_output / "valid").glob("*.jpg"))) > 0
    has_test = (dataset_output / "test").exists() and len(list((dataset_output / "test").glob("*.jpg"))) > 0

    if not has_valid and not has_test:
        warnings.append("No validation/test splits (only train)")
        print("   ⚠ Creating validation split from train data...")

        # Create a small validation split from train
        train_dir = dataset_output / "train"
        valid_dir = dataset_output / "valid"
        valid_dir.mkdir(exist_ok=True)

        train_json = train_dir / "_annotations.coco.json"
        if train_json.exists():
            with open(train_json, 'r') as f:
                train_data = json.load(f)

            # Take 1 image for validation (25% of 4 images)
            if len(train_data['images']) > 1:
                # Move last image to validation
                valid_img = train_data['images'].pop()
                valid_img_id = valid_img['id']

                # Move corresponding annotations
                valid_anns = [ann for ann in train_data['annotations'] if ann['image_id'] == valid_img_id]
                train_data['annotations'] = [ann for ann in train_data['annotations'] if ann['image_id'] != valid_img_id]

                # Create validation COCO file
                valid_data = {
                    'info': train_data['info'],
                    'licenses': train_data['licenses'],
                    'categories': train_data['categories'],
                    'images': [valid_img],
                    'annotations': valid_anns
                }

                # Save new splits
                with open(train_json, 'w') as f:
                    json.dump(train_data, f, indent=2)

                valid_json = valid_dir / "_annotations.coco.json"
                with open(valid_json, 'w') as f:
                    json.dump(valid_data, f, indent=2)

                # Move image file
                src_img = train_dir / valid_img['file_name']
                dst_img = valid_dir / valid_img['file_name']
                if src_img.exists():
                    shutil.move(src_img, dst_img)

                print(f"      Created validation split: 1 image, {len(valid_anns)} annotations")

    if warnings:
        print("   ⚠ Warnings:")
        for warning in warnings:
            print(f"      - {warning}")
    else:
        print("   ✓ All checks passed")

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"\nDataset location: {dataset_output}")
    print(f"\nNext steps:")
    print(f"  1. Review dataset in: {dataset_output}/")
    print(f"  2. Update training config with dataset path")
    print(f"  3. Start training: python scripts/train_sam3.py")
    print()

    return str(dataset_output)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Roboflow dataset for SAM3 training"
    )

    parser.add_argument(
        "--source",
        default="fuse-netrual-training-dataset",
        help="Path to downloaded Roboflow dataset"
    )

    parser.add_argument(
        "--output",
        default="sam3_datasets",
        help="Output directory for SAM3-formatted dataset"
    )

    parser.add_argument(
        "--name",
        default="fuse-cutout-detection",
        help="Dataset name"
    )

    args = parser.parse_args()

    prepare_sam3_dataset(args.source, args.output, args.name)


if __name__ == "__main__":
    main()
