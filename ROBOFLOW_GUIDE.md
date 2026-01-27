# Roboflow Annotation Pipeline Guide

Complete guide for downloading and validating your annotated dataset from Roboflow.

## Prerequisites

Install required Python packages:

```bash
pip install -r requirements.txt
```

## Step 1: Get Your Roboflow Information

You'll need these details from Roboflow:

### 1.1 Find Your API Key
1. Go to [Roboflow](https://app.roboflow.com)
2. Click your profile picture (top right)
3. Click **"Settings"** or **"Roboflow API"**
4. Copy your **API Key** (starts with letters/numbers like `abc123xyz...`)

### 1.2 Find Your Workspace and Project Names
Look at your Roboflow project URL:
```
https://app.roboflow.com/YOUR_WORKSPACE/YOUR_PROJECT/1
                              ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^
                              Workspace       Project
```

Example:
- URL: `https://app.roboflow.com/john-smith/fuse-cutout-detection/1`
- Workspace: `john-smith`
- Project: `fuse-cutout-detection`
- Version: `1`

### 1.3 Generate a Dataset Version (If Not Done Already)

If you haven't generated a version yet:

1. In Roboflow, open your project
2. Click **"Generate"** or **"Versions"** in left sidebar
3. Click **"Generate New Version"**
4. Choose preprocessing options (or use defaults)
5. Set train/val/test split:
   - Train: 70-80%
   - Valid: 10-20%
   - Test: 10%
6. Click **"Generate"**
7. Wait for processing (usually 1-2 minutes)
8. Note the version number (usually `1` for first version)

---

## Step 2: Download Your Dataset

Run the download script in **interactive mode** (easiest):

```bash
python scripts/roboflow_download.py
```

You'll be prompted to enter:
- API Key
- Workspace name
- Project name
- Version number (press Enter for default: 1)
- Output directory (press Enter for default: sam3_fuse_cutout_dataset)

### Alternative: Command Line Mode

If you prefer, you can pass all arguments directly:

```bash
python scripts/roboflow_download.py \
  --api-key YOUR_API_KEY \
  --workspace your-workspace \
  --project fuse-cutout-detection \
  --version 1 \
  --output sam3_fuse_cutout_dataset
```

### What Gets Downloaded

The script will create this structure:

```
sam3_fuse_cutout_dataset/
└── fuse-cutout-detection-1/    # (or your project name + version)
    ├── train/
    │   ├── _annotations.coco.json
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── ...
    └── test/
        ├── _annotations.coco.json
        └── ...
```

---

## Step 3: Validate Your Dataset

After downloading, validate that everything is correct:

```bash
python scripts/roboflow_validator.py
```

This will check:
- ✓ COCO JSON format is valid
- ✓ All required fields are present
- ✓ Images exist for all annotations
- ✓ Segmentation masks are valid
- ✓ Statistics (number of images, annotations, etc.)

### Example Output

```
==================================================================
COCO DATASET VALIDATOR
==================================================================

Dataset location: sam3_fuse_cutout_dataset/fuse-cutout-detection-1

Found splits: train, valid, test

Validating: _annotations.coco.json
--------------------------------------------------
  ✓ Valid COCO JSON structure
  ✓ Images: 5
  ✓ Annotations: 7
  ✓ Categories: 1

  Categories:
    - ID 0: fuse_cutout

  Annotation Statistics:
    - Avg annotations per image: 1.40
    - Avg annotation area: 12543.50 pixels²
    - Min area: 8234.00 pixels²
    - Max area: 18765.00 pixels²

  ✓ Found 5 image files in train/

==================================================================
✓ VALIDATION PASSED
==================================================================

Your dataset is ready for training!
```

---

## Step 4: Visualize Your Annotations

View your images with segmentation masks overlayed:

### Interactive Mode (Display on Screen)

```bash
python scripts/visualize_annotations.py
```

This will display images one by one. Press any key to see the next image, or press 'q' to quit.

### Save to Directory

To save visualizations instead of displaying them:

```bash
python scripts/visualize_annotations.py --output visualizations/
```

### Customize Number of Samples

By default, 5 samples per split are shown. To see more:

```bash
python scripts/visualize_annotations.py --num-samples 10
```

---

## Troubleshooting

### Error: "roboflow package not installed"

```bash
pip install roboflow
```

### Error: "Failed to access project"

- Check your workspace and project names match the URL exactly
- They are case-sensitive
- Don't include spaces or special characters

### Error: "Failed to get version"

- Make sure you've generated a dataset version in Roboflow
- Go to your project → "Generate" → Create version
- Check the version number (usually starts at 1)

### Error: "No COCO JSON files found"

- The download may have failed partway through
- Delete the output directory and try downloading again
- Check your internet connection

### Warning: "Images have no annotations"

- Some images in your dataset don't have any labeled objects
- This is okay for backgrounds, but verify it's intentional
- In Roboflow, check that all images were properly annotated

---

## Next Steps

Once your dataset is validated and visualized:

1. **Configure training**: Edit `configs/training_config.yaml`
2. **Start training**: Run `python scripts/train.py`
3. **Evaluate model**: Run `python scripts/evaluate.py`

---

## Quick Reference

```bash
# Download dataset
python scripts/roboflow_download.py

# Validate dataset
python scripts/roboflow_validator.py

# Visualize annotations (interactive)
python scripts/visualize_annotations.py

# Visualize and save to directory
python scripts/visualize_annotations.py --output visualizations/

# Get help
python scripts/roboflow_download.py --help
python scripts/roboflow_validator.py --help
python scripts/visualize_annotations.py --help
```

---

## Support

If you encounter issues:

1. Check you're using the correct API key and project names
2. Verify you've generated a dataset version in Roboflow
3. Make sure all dependencies are installed: `pip install -r requirements.txt`
4. Check the Roboflow documentation: https://docs.roboflow.com
