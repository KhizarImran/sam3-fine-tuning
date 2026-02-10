"""
Script to set up the new fuse neutrals dataset for SAM3 training
"""

import shutil
from pathlib import Path


def setup_fuse_neutrals_dataset():
    """Set up the Find fuse neutrals.v5i.coco dataset for SAM3 training."""

    print("=" * 70)
    print("SETTING UP FUSE NEUTRALS DATASET FOR SAM3")
    print("=" * 70)

    # Paths
    source_dir = Path("Find fuse neutrals.v5i.coco")
    target_dir = Path("sam3_datasets/fuse-neutrals")

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy all splits
    for split in ["train", "valid", "test"]:
        print(f"\nCopying {split} split...")
        source_split = source_dir / split
        target_split = target_dir / split

        if target_split.exists():
            print(f"  Removing existing {split} directory...")
            shutil.rmtree(target_split)

        print(f"  Copying from {source_split} to {target_split}...")
        shutil.copytree(source_split, target_split)

        # Count files
        images = list(target_split.glob("*.jpg")) + list(target_split.glob("*.png"))
        print(f"  Copied {len(images)} images")

        # Check annotation file
        ann_file = target_split / "_annotations.coco.json"
        if ann_file.exists():
            print(f"  Annotations: {ann_file.name}")
        else:
            print(f"  WARNING: No annotations found!")

    print("\n" + "=" * 70)
    print("DATASET SETUP COMPLETE")
    print("=" * 70)
    print(f"Dataset location: {target_dir.absolute()}")
    print("\nNext steps:")
    print("1. Update configs/fuse_neutrals_train.yaml")
    print("2. Run: python scripts/train_sam3.py -c configs/fuse_neutrals_train.yaml")


if __name__ == "__main__":
    setup_fuse_neutrals_dataset()
