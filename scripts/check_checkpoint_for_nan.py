#!/usr/bin/env python3
"""
Check if the checkpoint contains NaN or Inf values.
"""
import torch
from pathlib import Path

checkpoint_path = "experiments/fuse_cutout/checkpoints/checkpoint.pt"

print("=" * 80)
print("CHECKPOINT NaN/Inf DIAGNOSTIC")
print("=" * 80)

ckpt = torch.load(checkpoint_path, map_location="cpu")

print(f"\nCheckpoint keys: {list(ckpt.keys())}")

if 'model' in ckpt:
    model_state = ckpt['model']
    print(f"\nTotal parameters: {len(model_state)}")

    nan_params = []
    inf_params = []

    for name, param in model_state.items():
        if torch.isnan(param).any():
            nan_count = torch.isnan(param).sum().item()
            nan_params.append((name, nan_count, param.numel()))

        if torch.isinf(param).any():
            inf_count = torch.isinf(param).sum().item()
            inf_params.append((name, inf_count, param.numel()))

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if nan_params:
        print(f"\n❌ FOUND NaN VALUES IN {len(nan_params)} PARAMETERS:")
        for name, count, total in nan_params[:20]:  # Show first 20
            print(f"   {name}: {count}/{total} values are NaN ({100*count/total:.2f}%)")
        if len(nan_params) > 20:
            print(f"   ... and {len(nan_params) - 20} more parameters")
    else:
        print("\n✓ NO NaN values found in checkpoint")

    if inf_params:
        print(f"\n❌ FOUND Inf VALUES IN {len(inf_params)} PARAMETERS:")
        for name, count, total in inf_params[:20]:
            print(f"   {name}: {count}/{total} values are Inf ({100*count/total:.2f}%)")
        if len(inf_params) > 20:
            print(f"   ... and {len(inf_params) - 20} more parameters")
    else:
        print("\n✓ NO Inf values found in checkpoint")

    if not nan_params and not inf_params:
        print("\n✓ CHECKPOINT IS CLEAN - No NaN or Inf values")
        print("\nNaN during inference must be caused by:")
        print("  1. Numerical instability in forward pass")
        print("  2. Division by zero or log of negative number")
        print("  3. Input preprocessing issue")
    else:
        print("\n" + "=" * 80)
        print("DIAGNOSIS: CHECKPOINT IS CORRUPTED")
        print("=" * 80)
        print("\nThe checkpoint contains NaN/Inf values from training.")
        print("This happened because:")
        print("  1. Training diverged (loss became NaN)")
        print("  2. Learning rate too high")
        print("  3. Numerical instability during training")
        print("  4. Gradient explosion")
        print("\nSOLUTION: You need to RETRAIN the model")

if 'epoch' in ckpt:
    print(f"\n\nCheckpoint saved at epoch: {ckpt['epoch']}")

if 'loss' in ckpt:
    print(f"Training loss: {ckpt['loss']}")

print("\n" + "=" * 80)
