#!/usr/bin/env python3
"""
Verify SAM3 Setup and Inference Scripts

This script checks if all dependencies and files are properly set up
before running inference on EC2.
"""

import sys
from pathlib import Path
import importlib.util

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{text.center(70)}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓{Colors.NC} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠{Colors.NC} {text}")

def print_error(text):
    print(f"{Colors.RED}✗{Colors.NC} {text}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} found. Need Python 3.8+")
        return False

def check_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name

    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        print_success(f"{package_name} installed")
        return True
    else:
        print_error(f"{package_name} NOT installed")
        return False

def check_pytorch_cuda():
    """Check PyTorch and CUDA availability"""
    try:
        import torch
        print_success(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print_success(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print_warning("CUDA not available (will use CPU)")
            return True
    except ImportError:
        print_error("PyTorch NOT installed")
        return False

def check_sam3():
    """Check SAM3 installation"""
    sam3_path = Path(__file__).parent.parent / "sam3"

    if not sam3_path.exists():
        print_error(f"SAM3 directory not found at: {sam3_path}")
        return False

    print_success(f"SAM3 directory found: {sam3_path}")

    # Check SAM3 Python package
    sys.path.insert(0, str(sam3_path))
    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        print_success("SAM3 Python package importable")

        # Check BPE tokenizer
        bpe_path = sam3_path / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if bpe_path.exists():
            print_success(f"BPE tokenizer found: {bpe_path}")
        else:
            print_warning(f"BPE tokenizer not found at: {bpe_path}")
            print_warning("Text prompts may not work without tokenizer")

        return True
    except ImportError as e:
        print_error(f"Cannot import SAM3: {e}")
        print_error("Run: cd sam3 && pip install -e .")
        return False

def check_inference_scripts():
    """Check if inference scripts exist and are executable"""
    scripts_dir = Path(__file__).parent

    required_scripts = [
        "test_finetuned_model.py",
        "inference_sam3.py",
        "test_model_working.py",
        "test_model_correct.py"
    ]

    all_found = True
    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print_success(f"Found: {script}")
        else:
            print_error(f"Missing: {script}")
            all_found = False

    return all_found

def check_checkpoint():
    """Check for checkpoint files"""
    checkpoint_dirs = [
        Path("checkpoints"),
        Path("experiments/fuse_cutout/checkpoints"),
        Path("mlruns")
    ]

    found_checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("**/*.pt")) + list(checkpoint_dir.glob("**/*.pth"))
            if checkpoints:
                found_checkpoints.extend(checkpoints)

    if found_checkpoints:
        print_success(f"Found {len(found_checkpoints)} checkpoint(s):")
        for ckpt in found_checkpoints[:5]:  # Show first 5
            print(f"    {ckpt}")
        if len(found_checkpoints) > 5:
            print(f"    ... and {len(found_checkpoints) - 5} more")
    else:
        print_warning("No checkpoint files found")
        print_warning("You'll need a trained checkpoint to run inference")

    return len(found_checkpoints) > 0

def check_test_images():
    """Check for test images"""
    test_dirs = [
        Path("dataset/test/images"),
        Path("photos/test_fuse_neutral"),
        Path("sam3_datasets/fuse_cutout_dataset/test")
    ]

    found_images = []
    for test_dir in test_dirs:
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if images:
                found_images.append((test_dir, len(images)))

    if found_images:
        print_success("Found test images:")
        for dir_path, count in found_images:
            print(f"    {dir_path}: {count} images")
    else:
        print_warning("No test images found in common directories")

    return len(found_images) > 0

def print_next_steps(all_checks_passed):
    """Print next steps based on verification results"""
    print_header("NEXT STEPS")

    if all_checks_passed:
        print(f"{Colors.GREEN}All checks passed! ✓{Colors.NC}\n")
        print("You're ready to run inference. Try:")
        print(f"\n{Colors.YELLOW}# Quick test on single image{Colors.NC}")
        print("python scripts/test_finetuned_model.py \\")
        print("    --checkpoint path/to/checkpoint.pt \\")
        print("    --image path/to/image.jpg \\")
        print("    --text-prompt 'fuse cutout'")

        print(f"\n{Colors.YELLOW}# Batch processing{Colors.NC}")
        print("python scripts/test_finetuned_model.py \\")
        print("    --checkpoint path/to/checkpoint.pt \\")
        print("    --image-dir path/to/images/ \\")
        print("    --text-prompt 'fuse cutout' \\")
        print("    --threshold 0.5")

        print(f"\n{Colors.YELLOW}# Or use the shell script{Colors.NC}")
        print("bash scripts/run_inference.sh path/to/checkpoint.pt path/to/images/")

        print(f"\n{Colors.GREEN}Ready to push to GitHub and pull on EC2!{Colors.NC}")
    else:
        print(f"{Colors.RED}Some checks failed!{Colors.NC}\n")
        print("Please fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("  • Install missing packages: pip install torch numpy pillow matplotlib")
        print("  • Install SAM3: cd sam3 && pip install -e .")
        print("  • Check checkpoint location")

def main():
    print_header("SAM3 SETUP VERIFICATION")

    checks = []

    # 1. Python version
    print(f"{Colors.BLUE}[1/8]{Colors.NC} Checking Python version...")
    checks.append(check_python_version())

    # 2. Required packages
    print(f"\n{Colors.BLUE}[2/8]{Colors.NC} Checking required packages...")
    checks.append(check_package("torch"))
    checks.append(check_package("numpy"))
    checks.append(check_package("PIL", "pillow"))
    checks.append(check_package("matplotlib"))

    # 3. PyTorch CUDA
    print(f"\n{Colors.BLUE}[3/8]{Colors.NC} Checking PyTorch and CUDA...")
    checks.append(check_pytorch_cuda())

    # 4. SAM3
    print(f"\n{Colors.BLUE}[4/8]{Colors.NC} Checking SAM3 installation...")
    checks.append(check_sam3())

    # 5. Inference scripts
    print(f"\n{Colors.BLUE}[5/8]{Colors.NC} Checking inference scripts...")
    checks.append(check_inference_scripts())

    # 6. Checkpoints
    print(f"\n{Colors.BLUE}[6/8]{Colors.NC} Checking for model checkpoints...")
    check_checkpoint()  # Warning only, not critical

    # 7. Test images
    print(f"\n{Colors.BLUE}[7/8]{Colors.NC} Checking for test images...")
    check_test_images()  # Warning only, not critical

    # 8. Summary
    print(f"\n{Colors.BLUE}[8/8]{Colors.NC} Verification summary...")
    all_checks_passed = all(checks)

    print(f"\nPassed: {sum(checks)}/{len(checks)} critical checks")

    # Next steps
    print_next_steps(all_checks_passed)

    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
