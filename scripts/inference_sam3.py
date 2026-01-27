"""
SAM3 Inference Script

Test fine-tuned SAM3 model on images
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


def load_sam3_model(checkpoint_path):
    """Load fine-tuned SAM3 model"""
    print(f"Loading model from: {checkpoint_path}")

    # Add SAM3 to path
    sys.path.insert(0, "sam3")

    try:
        from sam3.build_sam import build_sam3_image_model
    except ImportError:
        print("ERROR: SAM3 not found. Run setup_sam3.bat first.")
        sys.exit(1)

    # Load model
    model = build_sam3_image_model(checkpoint_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("   Model loaded on GPU")
    else:
        print("   Model loaded on CPU (inference will be slow)")

    return model


def run_inference(model, image_path, text_prompt="fuse neutral", output_dir=None):
    """
    Run inference on a single image

    Args:
        model: Loaded SAM3 model
        image_path: Path to input image
        text_prompt: Text prompt for detection
        output_dir: Directory to save results (optional)
    """
    print(f"\nProcessing: {image_path}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Prepare input
    # TODO: Add proper SAM3 inference preprocessing
    # This is a placeholder - actual SAM3 inference API may differ

    print(f"   Image size: {image.size}")
    print(f"   Text prompt: '{text_prompt}'")

    # Run model inference
    with torch.no_grad():
        # TODO: Replace with actual SAM3 inference call
        # predictions = model.predict(image_np, text_prompt=text_prompt)
        print("   ⚠ Inference implementation pending - requires SAM3 API details")

    # Placeholder results
    print("   Results: [Inference not yet implemented]")

    # Save visualization if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save visualization
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_np)
        ax.set_title(f"SAM3 Inference: {Path(image_path).name}")
        ax.axis('off')

        save_path = output_path / f"{Path(image_path).stem}_result.jpg"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"   ✓ Saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 inference on images"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to fine-tuned model checkpoint (.pth file)"
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
        default="fuse neutral",
        help="Text prompt for detection (default: 'fuse neutral')"
    )

    parser.add_argument(
        "--output",
        default="inference_results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.image_dir:
        print("ERROR: Must specify either --image or --image-dir")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print("=" * 70)
    print("SAM3 INFERENCE")
    print("=" * 70)

    # Load model
    model = load_sam3_model(args.checkpoint)

    # Run inference
    if args.image:
        # Single image
        run_inference(model, args.image, args.text_prompt, args.output)

    elif args.image_dir:
        # Batch inference
        image_dir = Path(args.image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

        print(f"\nFound {len(image_files)} images")

        for image_file in image_files:
            run_inference(model, str(image_file), args.text_prompt, args.output)

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    main()
