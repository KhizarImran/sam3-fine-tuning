#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add SAM3 to path
sam3_path = Path(__file__).parent.parent / "sam3"
sys.path.insert(0, str(sam3_path))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--text-prompt", default="fuse cutout")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output", default="test_results_final")
    args = parser.parse_args()

    print("=" * 80)
    print("SAM3 INFERENCE - CORRECT VERSION")
    print("=" * 80)

    # CRITICAL: Need BPE tokenizer path
    bpe_path = Path(sam3_path) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        print(f"ERROR: BPE tokenizer not found at {bpe_path}")
        print("This is required for text prompts!")
        sys.exit(1)

    # Load model with checkpoint AND bpe_path
    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=args.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_mode=True,
        load_from_HF=False
    )
    print("✓ Model loaded")

    # Create processor
    processor = Sam3Processor(model, confidence_threshold=args.threshold)
    print(f"✓ Processor ready (threshold={args.threshold})")

    # Get images
    image_dir = Path(args.image_dir)
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    print(f"\nProcessing {len(image_paths)} images with prompt: '{args.text_prompt}'\n")

    # Create output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.name}")

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Run inference - CORRECT API
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=args.text_prompt, state=state)

        # Extract results
        boxes = state["boxes"]  # Already in pixel coordinates
        scores = state["scores"]
        masks = state["masks"]

        # Convert to numpy if needed
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        num_det = len(boxes)
        print(f"   ✓ {num_det} detections")
        if num_det > 0:
            print(f"   ✓ Avg score: {np.mean(scores):.3f}")

        # Visualize
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        for box, score in zip(boxes, scores):
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0), x1-x0, y1-y0,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x0, y0-5, f"{score:.2f}", color='lime', fontsize=12,
                   bbox=dict(facecolor='black', alpha=0.7))

        ax.set_title(f"{num_det} detections", fontsize=14)
        ax.axis('off')

        out_path = output_dir / f"{img_path.stem}_result.jpg"
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"   ✓ Saved: {out_path.name}")

        all_results.append({"path": str(img_path), "detections": num_det, "avg_score": np.mean(scores) if num_det > 0 else 0})

    # Summary
    total = sum(r["detections"] for r in all_results)
    print("\n" + "=" * 80)
    print(f"SUMMARY: {total} total detections across {len(all_results)} images")
    print("=" * 80)
    for r in all_results:
        print(f"  {Path(r['path']).name:40s} | {r['detections']:2d} detections | {r['avg_score']:.3f}")
    print(f"\n✓ Done! Results in: {output_dir}/")

if __name__ == "__main__":
    main()
