# Git Commit Checklist

## Files to Commit

### ✅ Core Fixes
- [x] `scripts/inference_sam3.py` - **FIXED** to use correct SAM3 API

### ✅ Documentation
- [x] `INFERENCE_GUIDE.md` - Complete inference guide
- [x] `FIXES_APPLIED.md` - Detailed fix documentation
- [x] `PUSH_TO_GITHUB_README.md` - Deployment instructions
- [x] `COMMIT_CHECKLIST.md` - This file

### ✅ Scripts (Existing - if changed)
- [x] `scripts/test_finetuned_model.py` - Main inference script
- [x] `scripts/test_model_correct.py` - Alternative implementation
- [x] `scripts/test_model_working.py` - Low threshold testing

### ✅ New Scripts
- [x] `scripts/run_inference.sh` - Quick inference shell script
- [x] `scripts/verify_setup.py` - Setup verification tool
- [x] `scripts/run_eval_on_test.sh` - Evaluation script

### ✅ Test Scripts
- [x] `test_before_push.bat` - Windows pre-push verification
- [x] `test_before_push.sh` - Linux pre-push verification

### ✅ Configuration
- [x] `.gitignore` - Git ignore file (excludes large data/models)

### ⚠️ Files to EXCLUDE (too large or not needed)
- [ ] `.claude/` - Claude Code metadata (in .gitignore)
- [ ] `dataset/` - Training data (too large)
- [ ] `photos/` - Image data (too large)
- [ ] `sam3/` - SAM3 source code (should be cloned separately)
- [ ] `validation_predictions_viz/` - Output data (too large)
- [ ] `checkpoints/` - Model files (too large, store separately)
- [ ] `*.pt`, `*.pth` - Checkpoint files (too large)

---

## Recommended Git Commands

### 1. Check what will be committed
```bash
git status
```

### 2. Add only the necessary files
```bash
# Add documentation
git add INFERENCE_GUIDE.md FIXES_APPLIED.md PUSH_TO_GITHUB_README.md COMMIT_CHECKLIST.md

# Add fixed scripts
git add scripts/inference_sam3.py

# Add new scripts
git add scripts/run_inference.sh scripts/verify_setup.py scripts/run_eval_on_test.sh

# Add test scripts (if not already tracked)
git add scripts/test_finetuned_model.py scripts/test_model_correct.py scripts/test_model_working.py

# Add test/verification scripts
git add test_before_push.bat test_before_push.sh

# Add gitignore
git add .gitignore

# Add planning docs if needed
git add SAM3_Fine_Tuning_Plan.txt RESEARCH_PAPER.txt
```

### 3. OR add everything (gitignore will exclude large files)
```bash
git add .
```

### 4. Check what's staged
```bash
git status
```

### 5. Commit with descriptive message
```bash
git commit -m "Fix SAM3 inference scripts and add comprehensive documentation

- Fixed inference_sam3.py to use correct SAM3 Processor API
- Replaced model.backbone.forward_text() with processor.set_text_prompt()
- Added build_sam3_image_model() with all required parameters
- Created comprehensive inference guide (INFERENCE_GUIDE.md)
- Added setup verification script (verify_setup.py)
- Added convenience shell script for quick inference (run_inference.sh)
- Added pre-push verification scripts for Windows and Linux
- Updated .gitignore to exclude large data/model files
- All scripts now production-ready for EC2 deployment

Verified against:
- SAM3 official documentation
- Context7 SAM3 API reference
- Working example scripts in codebase"
```

### 6. Push to GitHub
```bash
git push origin main
```

---

## After Push - EC2 Deployment

### 1. SSH into EC2
```bash
ssh ubuntu@your-ec2-instance
```

### 2. Pull changes
```bash
cd ~/sam3-fine-tuning
git pull origin main
```

### 3. Verify setup
```bash
source .venv/bin/activate
python scripts/verify_setup.py
```

### 4. Run inference
```bash
bash scripts/run_inference.sh checkpoints/best_model.pt dataset/test/images/
```

---

## Important Notes

### Files NOT to commit:
- Large model checkpoints (*.pt, *.pth)
- Training datasets (dataset/, photos/)
- Output visualizations (validation_predictions_viz/)
- SAM3 source code (should be separate submodule or clone)
- Virtual environment (.venv/)
- MLflow artifacts (mlruns/)

### How to handle large files:
1. Store model checkpoints on S3 or separate storage
2. Download on EC2 separately
3. Reference them with relative paths in scripts
4. Document checkpoint locations in README

### Checkpoint management:
```bash
# On EC2, download checkpoint from S3
aws s3 cp s3://your-bucket/checkpoints/best_model.pt checkpoints/

# Or use pre-existing checkpoint path
ls -lh checkpoints/
```

---

## Verification Before Push

Run this to verify everything is ready:

### Windows:
```cmd
test_before_push.bat
```

### Linux/Mac:
```bash
bash test_before_push.sh
```

This checks:
- ✓ Python environment
- ✓ Required packages
- ✓ SAM3 installation
- ✓ Script files
- ✓ Git status

---

## Quick Reference

### Current Status
```bash
git status --short
```

### What will be committed
```bash
git diff --cached --name-only
```

### Undo staging
```bash
git reset HEAD <file>
```

### View changes
```bash
git diff scripts/inference_sam3.py
```

---

**Ready to commit and push!** ✅
