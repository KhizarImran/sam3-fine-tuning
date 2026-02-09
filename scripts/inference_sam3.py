#!/usr/bin/env python3
"""
SAM3 Inference Script - FIXED VERSION

Test fine-tuned SAM3 model on images using correct SAM3 API
"""

import argparse
import json
from pathlib import Path
import sys

try:
    import torch
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError as e:
    print(f"ERROR: Required package not installed: {e}")
    print("Install with: pip install torch numpy pillow matplotlib")
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
    print("Make sure SAM3 is installed: cd sam3 && pip install -e .")
    sys.exit(1)


def load_sam3_model(checkpoint_path, device="cuda"):
    """Load fine-tuned SAM3 model using correct API"""
    print(f"Loading model from: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Build model with correct parameters
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
        eval_mode=True,
        load_from_HF=False,  # Don't download from HuggingFace
        enable_segmentation=True
    )

    print(f"   ✓ Model loaded on {device}")
    return model


def create_processor(model, device="cuda", confidence_threshold=0.5):
    """Create SAM3 processor for inference"""
    processor = Sam3Processor(
        model=model,
        resolution=1008,
        device=device,
        confidence_threshold=confidence_threshold
    )
    return processor


def run_inference(processor, image_path, text_prompt="fuse cutout", output_dir=None):
    """
    Run inference on a single image using SAM3 Processor API

    Args:
        processor: SAM3Processor instance
        image_path: Path to input image
        text_prompt: Text prompt for detection
        output_dir: Directory to save results (optional)

    Returns:
        dict: Results containing boxes, masks, scores
    """
    print(f"\nProcessing: {image_path}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    print(f"   Image size: {orig_w}x{orig_h}")
    print(f"   Text prompt: '{text_prompt}'")

    # Run inference using correct SAM3 API
    with torch.no_grad():
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=text_prompt, state=state)

    # Extract results
    boxes = state["boxes"]      # Shape: (N, 4), format [x0, y0, x1, y1]
    masks = state["masks"]      # Shape: (N, H, W)
    scores = state["scores"]    # Shape: (N,)

    # Convert to numpy
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    num_detections = len(boxes)
    print(f"   ✓ Found {num_detections} detections")

    if num_detections > 0:
        avg_score = np.mean(scores)
        print(f"   ✓ Average confidence: {avg_score:.3f}")

    results = {
        "image": image,
        "boxes": boxes,
        "masks": masks,
        "scores": scores,
        "image_size": (orig_w, orig_h),
        "num_detections": num_detections
    }

    # Save visualization if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create visualization
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(np.array(image))

        # Draw detections
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
            ax.text(
                x0, y0 - 5,
                f"{score:.2f}",
                color='lime',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2)
            )

        ax.set_title(
            f"SAM3: '{text_prompt}' | {num_detections} detections",
            fontsize=14,
            fontweight='bold'
        )
        ax.axis('off')

        save_path = output_path / f"{Path(image_path).stem}_result.jpg"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"   ✓ Saved to: {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 inference on images using fine-tuned model"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to fine-tuned model checkpoint (.pt file)"
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
        default="inference_results",
        help="Output directory for results"
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

    print("=" * 80)
    print("SAM3 INFERENCE - FIXED VERSION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Text prompt: '{args.text_prompt}'")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output}")
    print("=" * 80)

    # Load model
    model = load_sam3_model(args.checkpoint, device=args.device)

    # Create processor
    processor = create_processor(
        model,
        device=args.device,
        confidence_threshold=args.threshold
    )

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
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
        print(f"[{i}/{len(image_paths)}] {image_path.name}")

        try:
            results = run_inference(
                processor,
                image_path,
                args.text_prompt,
                args.output
            )
            all_results.append(results)

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    total_detections = sum(r["num_detections"] for r in all_results)
    avg_detections = total_detections / len(all_results) if all_results else 0

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Images processed: {len(all_results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    print(f"\n✓ Results saved to: {args.output}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
