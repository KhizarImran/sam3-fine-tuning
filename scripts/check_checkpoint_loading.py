#!/usr/bin/env python3
"""
Diagnostic script to verify if fine-tuned checkpoint weights are actually being loaded.
"""
import torch
from pathlib import Path
import sys

# Add SAM3 to path
sam3_path = Path(__file__).parent.parent / "sam3"
sys.path.insert(0, str(sam3_path))

from sam3 import build_sam3_image_model

print("=" * 80)
print("CHECKPOINT LOADING DIAGNOSTIC")
print("=" * 80)

bpe_path = Path(sam3_path) / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

print("\n1. Loading fine-tuned model with checkpoint...")
model_finetuned = build_sam3_image_model(
    bpe_path=str(bpe_path),
    checkpoint_path="experiments/fuse_cutout/checkpoints/checkpoint.pt",
    device="cpu",
    eval_mode=True,
    load_from_HF=False
)
print("   ✓ Fine-tuned model loaded")

print("\n2. Loading base model (no checkpoint)...")
model_base = build_sam3_image_model(
    bpe_path=str(bpe_path),
    device="cpu",
    eval_mode=True
)
print("   ✓ Base model loaded")

print("\n3. Comparing weights from transformer decoder...")

# Get weights from both models
finetuned_weight = model_finetuned.transformer.decoder.layers[0].self_attn.in_proj_weight
base_weight = model_base.transformer.decoder.layers[0].self_attn.in_proj_weight

# Compare statistics
ft_mean = finetuned_weight.mean().item()
ft_std = finetuned_weight.std().item()
base_mean = base_weight.mean().item()
base_std = base_weight.std().item()

print(f"\n   Fine-tuned weight statistics:")
print(f"     Mean: {ft_mean:.8f}")
print(f"     Std:  {ft_std:.8f}")

print(f"\n   Base model weight statistics:")
print(f"     Mean: {base_mean:.8f}")
print(f"     Std:  {base_std:.8f}")

# Check if they're the same
are_same = torch.allclose(finetuned_weight, base_weight, rtol=1e-5, atol=1e-8)

print("\n" + "=" * 80)
print("RESULT:")
print("=" * 80)

if are_same:
    print("❌ PROBLEM FOUND: Checkpoint weights NOT loaded!")
    print("   The model is using base weights, not your fine-tuned weights.")
    print("   This explains why you're getting 0 detections.")
    print("\n   Possible causes:")
    print("   - build_sam3_image_model() ignoring checkpoint_path parameter")
    print("   - Checkpoint file corrupted or wrong format")
    print("   - Weight loading logic has a bug")
else:
    print("✓ GOOD: Checkpoint weights loaded correctly!")
    print("  The fine-tuned weights are different from base weights.")
    print("\n  If you're still getting 0 detections, the issue is:")
    print("  - Text prompt mismatch (use: 'fuse neutral-xzg2yturox8qwldmiogk')")
    print("  - Sam3Processor inference pipeline not working properly")
    print("  - Need to use training evaluation code instead")

print("=" * 80)

# Additional check: inspect checkpoint file
print("\n4. Inspecting checkpoint file...")
ckpt = torch.load("experiments/fuse_cutout/checkpoints/checkpoint.pt", map_location="cpu")

print(f"   Checkpoint keys: {list(ckpt.keys())}")

if 'model' in ckpt:
    print(f"   Model state dict size: {len(ckpt['model'])} parameters")
if 'epoch' in ckpt:
    print(f"   Saved at epoch: {ckpt['epoch']}")
if 'optimizer' in ckpt:
    print(f"   Optimizer state saved: Yes")

print("\n✓ Diagnostic complete!")
