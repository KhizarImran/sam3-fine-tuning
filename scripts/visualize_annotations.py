"""
COCO Annotation Visualizer
Visualizes images with their segmentation masks overlayed
"""

import json
import os
import sys
from pathlib import Path
import argparse
import random

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"ERROR: Required package not installed: {e}")
    print("Install with: pip install opencv-python numpy Pillow")
    sys.exit(1)


def load_coco_annotations(json_path):
    """Load COCO format annotations"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create image id to annotations mapping
    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # Create image id to image info mapping
    images = {img['id']: img for img in data['images']}

    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in data['categories']}

    return images, img_annotations, categories


def draw_segmentation(image, segmentation, color, alpha=0.5):
    """Draw segmentation mask on image"""
    if not segmentation:
        return image

    overlay = image.copy()

    # Handle polygon segmentation (COCO format)
    if isinstance(segmentation, list):
        for seg in segmentation:
            if len(seg) >= 6:  # At least 3 points (x, y pairs)
                # Convert to numpy array of points
                points = np.array(seg).reshape(-1, 2).astype(np.int32)
                # Draw filled polygon
                cv2.fillPoly(overlay, [points], color)

    # Blend with original image
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw contour
    if isinstance(segmentation, list):
        for seg in segmentation:
            if len(seg) >= 6:
                points = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.polylines(result, [points], True, color, 2)

    return result


def visualize_sample(split_dir, json_file, output_dir=None, num_samples=5):
    """Visualize random samples from the dataset"""
    print(f"\nVisualizing samples from: {split_dir.name}")
    print("-" * 50)

    # Load annotations
    images, img_annotations, categories = load_coco_annotations(json_file)

    # Get random sample of images
    image_ids = list(images.keys())
    if len(image_ids) > num_samples:
        sample_ids = random.sample(image_ids, num_samples)
    else:
        sample_ids = image_ids

    # Generate random colors for each category
    category_colors = {}
    for cat_id in categories.keys():
        category_colors[cat_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    visualized_count = 0

    for img_id in sample_ids:
        img_info = images[img_id]
        img_filename = img_info['file_name']
        img_path = split_dir / img_filename

        if not img_path.exists():
            print(f"  ⚠ Image not found: {img_filename}")
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠ Failed to load: {img_filename}")
            continue

        # Get annotations for this image
        annotations = img_annotations.get(img_id, [])

        if not annotations:
            print(f"  ⚠ No annotations for: {img_filename}")
            continue

        # Draw each annotation
        for ann in annotations:
            category_id = ann.get('category_id')
            category_name = categories.get(category_id, 'unknown')
            segmentation = ann.get('segmentation', [])

            # Get color for this category
            color = category_colors.get(category_id, (0, 255, 0))

            # Draw segmentation
            image = draw_segmentation(image, segmentation, color, alpha=0.4)

            # Draw bounding box if available
            if 'bbox' in ann:
                bbox = ann['bbox']
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                # Add label
                label = f"{category_name}"
                cv2.putText(image, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save or display
        if output_dir:
            output_path = output_dir / f"visualized_{split_dir.name}_{img_filename}"
            cv2.imwrite(str(output_path), image)
            print(f"  ✓ Saved: {output_path.name}")
        else:
            # Display
            window_name = f"{split_dir.name} - {img_filename}"
            cv2.imshow(window_name, image)
            print(f"  ✓ Displaying: {img_filename} ({len(annotations)} annotations)")
            print(f"    Press any key to continue, 'q' to quit...")

            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == ord('q'):
                print("\n  Visualization stopped by user")
                break

        visualized_count += 1

    print(f"\n  Visualized {visualized_count} images")
    return visualized_count


def visualize_dataset(dataset_dir, output_dir=None, num_samples=5):
    """Visualize samples from the entire dataset"""
    print("=" * 70)
    print("COCO ANNOTATION VISUALIZER")
    print("=" * 70)

    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"\n✗ Dataset directory not found: {dataset_dir}")
        return

    # Find subdirectories
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"\n✗ No subdirectories found in {dataset_dir}")
        return

    actual_path = subdirs[0]
    print(f"\nDataset location: {actual_path}")

    # Create output directory if needed
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path}")
    else:
        output_path = None
        print("\nDisplaying images interactively (press any key to continue, 'q' to quit)")

    # Visualize each split
    splits = ['train', 'valid', 'test']
    total_visualized = 0

    for split in splits:
        split_dir = actual_path / split
        if not split_dir.exists():
            continue

        # Find COCO JSON
        json_files = list(split_dir.glob("_annotations.coco.json"))
        if not json_files:
            json_files = list(split_dir.glob("*.json"))

        if json_files:
            count = visualize_sample(split_dir, json_files[0], output_path, num_samples)
            total_visualized += count

    print("\n" + "=" * 70)
    print(f"✓ VISUALIZATION COMPLETE ({total_visualized} images)")
    print("=" * 70)

    if output_path:
        print(f"\nVisualized images saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize COCO format annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (display images)
  python scripts/visualize_annotations.py

  # Save to directory
  python scripts/visualize_annotations.py --output visualizations/

  # Visualize more samples
  python scripts/visualize_annotations.py --num-samples 10
        """
    )

    parser.add_argument(
        "--dataset-dir",
        default="sam3_fuse_cutout_dataset",
        help="Path to dataset directory (default: sam3_fuse_cutout_dataset)"
    )

    parser.add_argument(
        "--output",
        help="Output directory for visualized images (default: interactive display)"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize per split (default: 5)"
    )

    args = parser.parse_args()

    visualize_dataset(args.dataset_dir, args.output, args.num_samples)


if __name__ == "__main__":
    main()
