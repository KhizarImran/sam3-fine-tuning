#!/usr/bin/env python3
"""
Diagnostic script to see why inference returns 0 detections.
Shows intermediate scores before filtering.
"""
import sys
from pathlib import Path
import torch
from PIL import Image

# Add SAM3 to path
sam3_path = Path(__file__).parent.parent / "sam3"
sys.path.insert(0, str(sam3_path))

from sam3 import build_sam3_image_model

def diagnose_inference(checkpoint_path, image_path, text_prompt):
    print("=" * 80)
    print("INFERENCE SCORING DIAGNOSTIC")
    print("=" * 80)

    # Load model
    bpe_path = Path(sam3_path) / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_mode=True,
        load_from_HF=False
    )
    print("✓ Model loaded\n")

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path}")
    print(f"Size: {image.size}\n")

    # Prepare image
    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(model, confidence_threshold=0.0)  # No filtering!

    # Run inference
    state = processor.set_image(image)

    # MANUALLY call forward to intercept scores
    print(f"Text prompt: '{text_prompt}'")
    text_outputs = model.backbone.forward_text([text_prompt], device=model.device)

    # Create find stage
    from sam3.model.data_misc import FindStage
    find_stage = FindStage(
        img_ids=torch.tensor([0], device=model.device, dtype=torch.long),
        text_ids=torch.tensor([0], device=model.device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )

    # Forward through model
    backbone_out = state["backbone_out"]
    backbone_out.update(text_outputs)

    # Get dummy geometric prompt (no box prompts)
    geometric_prompt = model._get_dummy_prompt()

    outputs = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_stage,
        geometric_prompt=geometric_prompt,
        find_target=None,
    )

    # Extract and analyze scores
    out_logits = outputs["pred_logits"][0]  # [num_queries, 1]
    presence_logit = outputs["presence_logit_dec"]  # [batch, 1]

    out_probs = out_logits.sigmoid()
    presence_score = presence_logit.sigmoid()

    combined_score = (out_probs * presence_score).squeeze(-1)

    print("\n" + "=" * 80)
    print("SCORE ANALYSIS")
    print("=" * 80)

    print(f"\n1. Detection Logits (pred_logits):")
    print(f"   Shape: {out_logits.shape}")
    print(f"   Raw logits: min={out_logits.min().item():.4f}, max={out_logits.max().item():.4f}, mean={out_logits.mean().item():.4f}")
    print(f"   After sigmoid: min={out_probs.min().item():.6f}, max={out_probs.max().item():.6f}, mean={out_probs.mean().item():.6f}")
    print(f"   Top 5 scores: {torch.topk(out_probs.squeeze(), 5).values.tolist()}")

    print(f"\n2. Presence Score (presence_logit_dec):")
    print(f"   Shape: {presence_logit.shape}")
    print(f"   Raw logit: {presence_logit.item():.4f}")
    print(f"   After sigmoid: {presence_score.item():.6f}")

    print(f"\n3. Combined Score (detection × presence):")
    print(f"   min={combined_score.min().item():.6f}, max={combined_score.max().item():.6f}, mean={combined_score.mean().item():.6f}")
    print(f"   Top 5 scores: {torch.topk(combined_score, 5).values.tolist()}")

    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    for threshold in [0.01, 0.05, 0.1, 0.3, 0.5]:
        keep = combined_score > threshold
        num_kept = keep.sum().item()
        print(f"   Threshold {threshold:.2f}: {num_kept} detections would pass")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if presence_score.item() < 0.1:
        print("\n❌ PROBLEM: Presence score is very low!")
        print(f"   Presence score: {presence_score.item():.6f}")
        print("   This suppresses ALL detection scores.")
        print("\n   CAUSE: The presence decoder head wasn't properly trained.")
        print("   With small datasets, this head often doesn't converge.")
        print("\n   SOLUTIONS:")
        print("   1. Use very low threshold (0.01-0.05)")
        print("   2. Retrain with more emphasis on presence supervision")
        print("   3. Modify Sam3Processor to ignore presence score")
    elif out_probs.max().item() < 0.3:
        print("\n❌ PROBLEM: Detection scores are low!")
        print(f"   Max detection score: {out_probs.max().item():.4f}")
        print("\n   CAUSE: Model not confident in detections.")
        print("   SOLUTIONS:")
        print("   1. Check if text prompt matches training")
        print("   2. Retrain model longer")
        print("   3. Use lower threshold")
    elif combined_score.max().item() > 0.05:
        print("\n✓ GOOD: Scores look reasonable!")
        print(f"   Max combined score: {combined_score.max().item():.4f}")
        print(f"   Use threshold ≤ {combined_score.max().item():.4f} to get detections")
    else:
        print("\n❓ UNCLEAR: Both scores are low.")
        print("   Model may not be recognizing the object.")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--text-prompt", default="fuse neutral")
    args = parser.parse_args()

    diagnose_inference(args.checkpoint, args.image, args.text_prompt)
