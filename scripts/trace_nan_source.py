#!/usr/bin/env python3
"""
Trace where NaN values first appear during inference.
"""
import sys
from pathlib import Path
import torch
from PIL import Image

# Add SAM3 to path
sam3_path = Path(__file__).parent.parent / "sam3"
sys.path.insert(0, str(sam3_path))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def check_for_nan(tensor, name):
    """Check if tensor contains NaN and report"""
    if tensor is None:
        print(f"  {name}: None")
        return False

    if isinstance(tensor, dict):
        has_nan = False
        for k, v in tensor.items():
            if check_for_nan(v, f"{name}[{k}]"):
                has_nan = True
        return has_nan

    if isinstance(tensor, (list, tuple)):
        has_nan = False
        for i, v in enumerate(tensor):
            if check_for_nan(v, f"{name}[{i}]"):
                has_nan = True
        return has_nan

    if isinstance(tensor, torch.Tensor):
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan or has_inf:
            nan_count = torch.isnan(tensor).sum().item() if has_nan else 0
            inf_count = torch.isinf(tensor).sum().item() if has_inf else 0
            print(f"  ❌ {name}: shape={list(tensor.shape)}, NaN={nan_count}, Inf={inf_count}")
            print(f"     min={tensor[~torch.isnan(tensor)].min().item() if not has_nan or nan_count < tensor.numel() else 'all NaN'}, "
                  f"max={tensor[~torch.isnan(tensor)].max().item() if not has_nan or nan_count < tensor.numel() else 'all NaN'}")
            return True
        else:
            print(f"  ✓ {name}: shape={list(tensor.shape)}, min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")
            return False

    return False

def trace_inference(checkpoint_path, image_path, text_prompt):
    print("=" * 80)
    print("TRACING NaN SOURCE IN INFERENCE")
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

    processor = Sam3Processor(model, confidence_threshold=0.0)

    # STEP 1: Check image preprocessing
    print("=" * 80)
    print("STEP 1: IMAGE PREPROCESSING")
    print("=" * 80)

    from torchvision.transforms import v2
    image_tensor = v2.functional.to_image(image).to(model.device)
    print(f"\n1.1 After to_image:")
    check_for_nan(image_tensor, "image_tensor")

    image_tensor = processor.transform(image_tensor).unsqueeze(0)
    print(f"\n1.2 After transform + normalize:")
    check_for_nan(image_tensor, "image_tensor")

    # STEP 2: Check backbone forward_image
    print("\n" + "=" * 80)
    print("STEP 2: BACKBONE IMAGE ENCODING")
    print("=" * 80)

    with torch.no_grad():
        backbone_out = model.backbone.forward_image(image_tensor)

    print("\nBackbone output keys:", list(backbone_out.keys()))
    for key in backbone_out.keys():
        if key != "sam2_backbone_out":  # Skip nested dict for now
            check_for_nan(backbone_out[key], f"backbone_out[{key}]")

    # STEP 3: Check text encoding
    print("\n" + "=" * 80)
    print("STEP 3: TEXT ENCODING")
    print("=" * 80)

    print(f"\nText prompt: '{text_prompt}'")
    with torch.no_grad():
        text_outputs = model.backbone.forward_text([text_prompt], device=model.device)

    print("\nText output keys:", list(text_outputs.keys()))
    for key in text_outputs.keys():
        check_for_nan(text_outputs[key], f"text_outputs[{key}]")

    # STEP 4: Check forward_grounding
    print("\n" + "=" * 80)
    print("STEP 4: GROUNDING FORWARD PASS")
    print("=" * 80)

    backbone_out.update(text_outputs)
    geometric_prompt = model._get_dummy_prompt()

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

    with torch.no_grad():
        outputs = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

    print("\nGrounding output keys:", list(outputs.keys()))
    for key in outputs.keys():
        check_for_nan(outputs[key], f"outputs[{key}]")

    # STEP 5: Final diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if check_for_nan(outputs['pred_logits'], "pred_logits_final_check"):
        print("\n❌ PROBLEM: pred_logits contains NaN")

    if check_for_nan(outputs['presence_logit_dec'], "presence_logit_final_check"):
        print("\n❌ PROBLEM: presence_logit_dec contains NaN")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--text-prompt", default="fuse neutral-xzg2yturox8qwldmiogk")
    args = parser.parse_args()

    trace_inference(args.checkpoint, args.image, args.text_prompt)
