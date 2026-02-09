#!/usr/bin/env python3
"""
SAM3 Inference Script for Fuse Neutral Detection

Tests the trained checkpoint on images and outputs visualizations with bounding boxes.
Ready to use after training completes!
"""
import sys
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from datetime import datetime

# Add SAM3 to path
sam3_path = Path(__file__).parent.parent / "sam3"
sys.path.insert(0, str(sam3_path))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def test_checkpoint_inference(
    checkpoint_path,
    image_dir,
    text_prompt="fuse neutral",
    confidence_threshold=0.05,
    output_dir="test_results",
    save_json=True
):
    """
    Run inference on images using trained SAM3 checkpoint

    Args:
        checkpoint_path: Path to trained checkpoint (.pt file)
        image_dir: Directory containing test images
        text_prompt: Text prompt for detection (default: "fuse neutral")
        confidence_threshold: Minimum confidence score (default: 0.05)
        output_dir: Where to save results
        save_json: Whether to save results as JSON
    """

    print("=" * 80)
    print("SAM3 FUSE NEUTRAL DETECTION - INFERENCE")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Text Prompt: '{text_prompt}'")
    print(f"Confidence Threshold: {confidence_threshold}")
    print("=" * 80)

    # Verify checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("\nMake sure training has completed and checkpoint was saved.")
        sys.exit(1)

    # Verify BPE tokenizer exists
    bpe_path = Path(sam3_path) / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        print(f"❌ ERROR: BPE tokenizer not found at {bpe_path}")
        print("This is required for text prompts!")
        sys.exit(1)

    # Load model - first build base model, then load training checkpoint
    print(f"\n📦 Loading model from checkpoint...")
    try:
        import torch
        from iopath.common.file_io import g_pathmgr

        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ Using GPU: {gpu_name}")
        else:
            print("⚠ WARNING: No GPU detected, inference will be slow on CPU")

        # Build model with HuggingFace pretrained weights first
        model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            checkpoint_path=None,  # Don't use checkpoint_path here
            device="cpu",  # Load on CPU first
            eval_mode=True,
            load_from_HF=True  # Load pretrained weights from HuggingFace
        )
        print("✓ Base model loaded with pretrained weights")

        # Now load the fine-tuned checkpoint on top
        with g_pathmgr.open(str(checkpoint_path), "rb") as f:
            ckpt = torch.load(f, map_location="cpu")

        # Extract model state_dict
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Load the fine-tuned weights (strict=False to allow missing keys from frozen layers)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"✓ Fine-tuned weights loaded")
        print(f"   Missing keys: {len(missing_keys)} (frozen/pretrained layers)")
        print(f"   Unexpected keys: {len(unexpected_keys)}")

        # Move to GPU after loading
        if device == "cuda":
            model = model.cuda()

        print("✓ Model ready for inference")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create processor
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    print(f"✓ Processor initialized (threshold={confidence_threshold})")

    # Get test images
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"❌ ERROR: Image directory not found: {image_dir}")
        sys.exit(1)

    image_paths = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.JPG"))
    )

    if not image_paths:
        print(f"❌ ERROR: No images found in {image_dir}")
        print("Looking for: *.jpg, *.png, *.jpeg, *.JPG")
        sys.exit(1)

    print(f"\n📸 Found {len(image_paths)} test images")
    print("=" * 80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference on all images
    all_results = []
    total_detections = 0

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {img_path.name}")

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            img_width, img_height = image.size
            print(f"   Image size: {img_width}x{img_height}")

            # Run inference using correct Sam3Processor API
            state = processor.set_image(image)
            state = processor.set_text_prompt(prompt=text_prompt, state=state)

            # Extract results
            boxes = state["boxes"]  # [N, 4] in pixel coords [x0, y0, x1, y1]
            scores = state["scores"]  # [N] confidence scores
            masks = state["masks"]  # [N, H, W] boolean masks

            # Convert to numpy for processing
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            num_detections = len(boxes)
            total_detections += num_detections

            if num_detections == 0:
                print(f"   ⚠ No detections found")
            else:
                print(f"   ✓ Found {num_detections} fuse neutral(s)")
                print(f"   ✓ Score range: {scores.min():.3f} - {scores.max():.3f}")
                print(f"   ✓ Average score: {scores.mean():.3f}")

            # Visualize results
            fig, ax = plt.subplots(1, figsize=(14, 10))
            ax.imshow(image)

            # Draw bounding boxes
            for box, score in zip(boxes, scores):
                x0, y0, x1, y1 = box
                width = x1 - x0
                height = y1 - y0

                # Draw rectangle
                rect = patches.Rectangle(
                    (x0, y0), width, height,
                    linewidth=3,
                    edgecolor='lime',
                    facecolor='none'
                )
                ax.add_patch(rect)

                # Add score label
                label = f"Fuse Neutral: {score:.2f}"
                ax.text(
                    x0, y0 - 5,
                    label,
                    color='lime',
                    fontsize=12,
                    weight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2)
                )

            # Title with detection count
            title = f"{img_path.name}\n{num_detections} Fuse Neutral(s) Detected"
            ax.set_title(title, fontsize=16, weight='bold')
            ax.axis('off')

            # Save visualization
            out_path = output_dir / f"{img_path.stem}_detected.jpg"
            plt.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
            print(f"   ✓ Saved visualization: {out_path.name}")

            # Store results
            detections_list = []
            for box, score in zip(boxes, scores):
                x0, y0, x1, y1 = box
                detections_list.append({
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "score": float(score),
                    "category": text_prompt
                })

            result = {
                "image_name": img_path.name,
                "image_path": str(img_path),
                "image_size": [img_width, img_height],
                "num_detections": num_detections,
                "detections": detections_list,
                "avg_score": float(scores.mean()) if num_detections > 0 else 0.0,
                "max_score": float(scores.max()) if num_detections > 0 else 0.0
            }
            all_results.append(result)

        except Exception as e:
            print(f"   ❌ ERROR processing image: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Total Images Processed: {len(all_results)}")
    print(f"Total Fuse Neutrals Detected: {total_detections}")
    print(f"Average Detections per Image: {total_detections / len(all_results):.2f}")
    print("\nPer-Image Results:")
    print("-" * 80)

    for result in all_results:
        det_count = result['num_detections']
        avg_score = result['avg_score']
        status = "✓" if det_count > 0 else "○"
        print(f"  {status} {result['image_name']:40s} | {det_count:2d} detections | Avg: {avg_score:.3f}")

    # Save results as JSON
    if save_json:
        json_path = output_dir / "inference_results.json"
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "checkpoint": str(checkpoint_path),
                "text_prompt": text_prompt,
                "confidence_threshold": confidence_threshold,
                "total_images": len(all_results),
                "total_detections": total_detections
            },
            "results": all_results
        }

        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n✓ Results saved to: {json_path}")

    print(f"\n✓ Visualizations saved to: {output_dir}/")
    print("=" * 80)

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test SAM3 trained checkpoint on fuse neutral detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on default images
  python scripts/test_fuse_neutral.py

  # Test with custom checkpoint and images
  python scripts/test_fuse_neutral.py --checkpoint experiments/fuse_cutout/checkpoints/checkpoint.pt --images photos/test_fuse_neutral

  # Lower threshold to see more detections
  python scripts/test_fuse_neutral.py --threshold 0.01

  # Use different text prompt
  python scripts/test_fuse_neutral.py --prompt "fuse cutout"
        """
    )

    parser.add_argument(
        "--checkpoint",
        default="experiments/fuse_cutout/checkpoints/checkpoint.pt",
        help="Path to trained checkpoint (default: experiments/fuse_cutout/checkpoints/checkpoint.pt)"
    )

    parser.add_argument(
        "--images",
        default="photos/test_fuse_neutral",
        help="Directory containing test images (default: photos/test_fuse_neutral)"
    )

    parser.add_argument(
        "--prompt",
        default="fuse neutral",
        help="Text prompt for detection (default: 'fuse neutral')"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Confidence threshold (default: 0.05, try 0.01 for more detections)"
    )

    parser.add_argument(
        "--output",
        default="test_results",
        help="Output directory for results (default: test_results)"
    )

    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't save results as JSON"
    )

    args = parser.parse_args()

    # Run inference
    test_checkpoint_inference(
        checkpoint_path=args.checkpoint,
        image_dir=args.images,
        text_prompt=args.prompt,
        confidence_threshold=args.threshold,
        output_dir=args.output,
        save_json=not args.no_json
    )


if __name__ == "__main__":
    main()
