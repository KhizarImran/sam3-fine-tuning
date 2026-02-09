#!/usr/bin/env python3
"""
Test Fine-tuned SAM3 Model

Run inference on test images using the fine-tuned checkpoint
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

try:
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError as e:
    print(f"ERROR: Required package not installed: {e}")
    print("Install with: uv pip install torch numpy pillow matplotlib")
    sys.exit(1)

# Add SAM3 to path
sam3_path = Path(__file__).parent.parent / "sam3"
if sam3_path.exists():
    sys.path.insert(0, str(sam3_path))
else:
    print(f"ERROR: SAM3 not found at {sam3_path}")
    sys.exit(1)

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"ERROR: Cannot import SAM3: {e}")
    print("Make sure SAM3 is installed: cd sam3 && uv pip install -e .")
    sys.exit(1)


def load_model(checkpoint_path, device="cuda"):
    """Load fine-tuned SAM3 model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Build model with checkpoint
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
        eval_mode=True,
        load_from_HF=False
    )

    print(f"   ✓ Model loaded on {device}")
    return model


def create_processor(model, device="cuda", confidence_threshold=0.5, resolution=1008):
    """Create SAM3 processor for inference."""
    processor = Sam3Processor(
        model=model,
        resolution=resolution,
        device=device,
        confidence_threshold=confidence_threshold
    )
    return processor


def run_inference(processor, image_path, text_prompt="fuse cutout"):
    """
    Run inference on a single image.

    Args:
        processor: SAM3Processor instance
        image_path: Path to input image
        text_prompt: Text prompt for detection

    Returns:
        dict: Results containing boxes, masks, scores
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # Run inference
    with torch.no_grad():
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=text_prompt, state=state)

    # Extract results
    boxes = state["boxes"]      # Shape: (N, 4), format [x0, y0, x1, y1]
    masks = state["masks"]      # Shape: (N, H, W)
    scores = state["scores"]    # Shape: (N,)

    # Convert to numpy for easier handling
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    return {
        "image": image,
        "boxes": boxes,
        "masks": masks,
        "scores": scores,
        "image_size": (orig_w, orig_h),
        "num_detections": len(boxes)
    }


def visualize_results(results, image_path, output_dir, text_prompt="fuse cutout"):
    """
    Create visualization with bounding boxes and save to file.

    Args:
        results: Inference results dict
        image_path: Original image path
        output_dir: Directory to save visualization
        text_prompt: Text prompt used for detection
    """
    image = results["image"]
    boxes = results["boxes"]
    scores = results["scores"]

    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(np.array(image))

    # Draw each detection
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0

        # Draw bounding box
        rect = patches.Rectangle(
            (x0, y0), width, height,
            linewidth=2,
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add score label
        label = f"{score:.2f}"
        ax.text(
            x0, y0 - 5,
            label,
            color='lime',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2)
        )

    # Add title
    ax.set_title(
        f"SAM3 Fine-tuned: '{text_prompt}' | {len(boxes)} detections",
        fontsize=14,
        fontweight='bold'
    )
    ax.axis('off')

    # Save
    output_path = Path(output_dir) / f"{Path(image_path).stem}_result.jpg"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    return output_path


def save_results_json(all_results, output_dir):
    """Save all results to JSON file."""
    output_path = Path(output_dir) / "results.json"

    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for result in all_results:
        json_result = {
            "image_path": result["image_path"],
            "num_detections": result["num_detections"],
            "boxes": result["boxes"].tolist() if isinstance(result["boxes"], np.ndarray) else result["boxes"],
            "scores": result["scores"].tolist() if isinstance(result["scores"], np.ndarray) else result["scores"],
            "image_size": result["image_size"]
        }
        json_results.append(json_result)

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    return output_path


def print_summary(all_results):
    """Print summary statistics."""
    total_images = len(all_results)
    total_detections = sum(r["num_detections"] for r in all_results)
    avg_detections = total_detections / total_images if total_images > 0 else 0

    # Collect all scores
    all_scores = []
    for r in all_results:
        if len(r["scores"]) > 0:
            all_scores.extend(r["scores"])

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total images processed: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")

    if all_scores:
        print(f"Average confidence score: {np.mean(all_scores):.3f}")
        print(f"Min confidence score: {np.min(all_scores):.3f}")
        print(f"Max confidence score: {np.max(all_scores):.3f}")

    print("\nPer-image breakdown:")
    print("-" * 80)
    for result in all_results:
        img_name = Path(result["image_path"]).name
        num_det = result["num_detections"]
        avg_score = np.mean(result["scores"]) if len(result["scores"]) > 0 else 0.0
        print(f"  {img_name:40s} | {num_det:2d} detections | avg score: {avg_score:.3f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned SAM3 model on images"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to fine-tuned checkpoint (.pt file)"
    )

    parser.add_argument(
        "--image",
        help="Path to single image for inference"
    )

    parser.add_argument(
        "--image-dir",
        help="Path to directory of images for batch inference"
    )

    parser.add_argument(
        "--text-prompt",
        default="fuse cutout",
        help="Text prompt for detection (default: 'fuse cutout')"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )

    parser.add_argument(
        "--output",
        default="test_results",
        help="Output directory for results (default: test_results)"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.image_dir:
        print("ERROR: Must specify either --image or --image-dir")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SAM3 FINE-TUNED MODEL TESTING")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Text prompt: '{args.text_prompt}'")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Device: {args.device}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load model
    model = load_model(args.checkpoint, device=args.device)

    # Create processor
    processor = create_processor(
        model,
        device=args.device,
        confidence_threshold=args.threshold
    )

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths.extend(image_dir.glob("*.jpg"))
        image_paths.extend(image_dir.glob("*.jpeg"))
        image_paths.extend(image_dir.glob("*.png"))
        image_paths = sorted(image_paths)

    if not image_paths:
        print("ERROR: No images found")
        sys.exit(1)

    print(f"\nFound {len(image_paths)} images to process\n")

    # Run inference on all images
    all_results = []

    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")

        try:
            # Run inference
            results = run_inference(processor, image_path, args.text_prompt)

            # Add image path to results
            results["image_path"] = str(image_path)

            print(f"   ✓ Found {results['num_detections']} detections")

            if results["num_detections"] > 0:
                avg_score = np.mean(results["scores"])
                print(f"   ✓ Average confidence: {avg_score:.3f}")

            # Visualize and save
            output_path = visualize_results(
                results, image_path, output_dir, args.text_prompt
            )
            print(f"   ✓ Saved visualization: {output_path.name}")

            all_results.append(results)

        except Exception as e:
            print(f"   ✗ Error processing image: {e}")
            continue

    # Save results to JSON
    json_path = save_results_json(all_results, output_dir)
    print(f"\n✓ Saved results JSON: {json_path}")

    # Print summary
    print_summary(all_results)

    print(f"\n✓ All results saved to: {output_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
