#!/usr/bin/env python3
"""
Test Fine-tuned SAM3 Model - CORRECTED VERSION
Uses SAM3's actual inference pipeline (same as training evaluation)
"""

import sys
import argparse
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add SAM3 to path
sam3_path = Path(__file__).parent.parent / "sam3"
sys.path.insert(0, str(sam3_path))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image import Sam3ForImage


def load_model(checkpoint_path, device="cuda"):
    """Load fine-tuned SAM3 model."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint directly
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Build model
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
        eval_mode=True
    )

    model.eval()
    print(f"✓ Model loaded on {device}")
    return model


def run_inference_native(model, image_path, confidence_threshold=0.5):
    """
    Run inference using SAM3's native forward pass (same as training).
    """
    from sam3.model.sam3_image_processor import Sam3ForImageProcessor
    from sam3.data.datasets.image_datasets.collator import SparseImageCollator

    # Load image
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # Prepare input in SAM3's expected format
    # Resize to model resolution (1008)
    image_resized = image.resize((1008, 1008))
    image_np = np.array(image_resized).astype(np.float32) / 255.0

    # Convert to tensor [1, 3, H, W]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    # Prepare batch dict (mimicking training data format)
    batch = {
        "image": image_tensor,
        "original_size": torch.tensor([[orig_h, orig_w]]),
        "resized_size": torch.tensor([[1008, 1008]]),
    }

    # Run model forward
    with torch.no_grad():
        outputs = model(batch)

    # Extract predictions
    # Outputs should contain: boxes, scores, labels
    if isinstance(outputs, dict):
        pred_boxes = outputs.get("pred_boxes", outputs.get("boxes", None))
        pred_scores = outputs.get("pred_logits", outputs.get("scores", None))

        if pred_scores is not None and pred_scores.dim() > 2:
            # Convert logits to scores
            pred_scores = pred_scores.sigmoid().max(dim=-1)[0]

        if pred_boxes is not None:
            pred_boxes = pred_boxes[0]  # Remove batch dim
        if pred_scores is not None:
            pred_scores = pred_scores[0] if pred_scores.dim() > 1 else pred_scores
    else:
        # Outputs might be a tuple
        pred_boxes = outputs[0] if len(outputs) > 0 else None
        pred_scores = outputs[1] if len(outputs) > 1 else None

    # Filter by confidence threshold
    if pred_boxes is not None and pred_scores is not None:
        keep = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]

        # Convert to numpy
        boxes = pred_boxes.cpu().numpy()
        scores = pred_scores.cpu().numpy()

        # Convert from normalized [0,1] to pixel coordinates if needed
        if boxes.max() <= 1.0:
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h
    else:
        boxes = np.array([])
        scores = np.array([])

    return {
        "image": image,
        "boxes": boxes,
        "scores": scores,
        "image_size": (orig_w, orig_h),
        "num_detections": len(boxes)
    }


def visualize_and_save(results, image_path, output_dir):
    """Visualize predictions and save."""
    image = results["image"]
    boxes = results["boxes"]
    scores = results["scores"]

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(np.array(image))

    for box, score in zip(boxes, scores):
        if len(box) == 4:
            x0, y0, x1, y1 = box
            width = x1 - x0
            height = y1 - y0

            rect = patches.Rectangle(
                (x0, y0), width, height,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(
                x0, y0 - 5, f"{score:.2f}",
                color='lime', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2)
            )

    ax.set_title(f"Detections: {len(boxes)}", fontsize=14, fontweight='bold')
    ax.axis('off')

    output_path = Path(output_dir) / f"{Path(image_path).stem}_result.jpg"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="test_results_correct")
    args = parser.parse_args()

    print("=" * 80)
    print("SAM3 INFERENCE - CORRECTED VERSION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold: {args.threshold}")
    print("=" * 80)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    # Get images
    image_dir = Path(args.image_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    print(f"\nFound {len(image_paths)} images\n")

    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    all_results = []

    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")

        try:
            results = run_inference_native(model, image_path, args.threshold)
            results["image_path"] = str(image_path)

            print(f"   ✓ Found {results['num_detections']} detections")
            if results['num_detections'] > 0:
                print(f"   ✓ Avg score: {np.mean(results['scores']):.3f}")

            output_path = visualize_and_save(results, image_path, output_dir)
            print(f"   ✓ Saved: {output_path.name}")

            all_results.append(results)

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    total_det = sum(r['num_detections'] for r in all_results)
    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(all_results)} images, {total_det} total detections")
    print("=" * 80)

    for r in all_results:
        img_name = Path(r['image_path']).name
        print(f"  {img_name:40s} | {r['num_detections']:2d} detections")

    print(f"\n✓ Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
