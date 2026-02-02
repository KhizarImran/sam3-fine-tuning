#!/bin/bash
# Run SAM3 evaluation on test images using the training pipeline

# First, create a minimal COCO annotation file for test images
python << 'EOF'
import json
from pathlib import Path
from PIL import Image

test_dir = Path("photos/test_fuse_neutral")
images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "fuse-cutout"}]
}

for idx, img_path in enumerate(images, 1):
    img = Image.open(img_path)
    w, h = img.size
    coco_data["images"].append({
        "id": idx,
        "file_name": img_path.name,
        "width": w,
        "height": h
    })

# Save dummy annotations (empty, just for evaluation format)
output_dir = Path("sam3_datasets/test_images")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "_annotations.coco.json", 'w') as f:
    json.dump(coco_data, f, indent=2)

# Copy images
import shutil
for img_path in images:
    shutil.copy(img_path, output_dir / img_path.name)

print(f"Created test dataset with {len(images)} images at {output_dir}")
EOF

# Now run evaluation using SAM3's training pipeline
echo "Running SAM3 evaluation..."
python sam3/sam3/train/train.py \
  --config-name fuse_cutout_train \
  trainer.skip_training=true \
  trainer.do_eval_only=true \
  paths.roboflow_vl_100_root=sam3_datasets \
  roboflow_train.supercategory=test_images \
  model.checkpoint_path=experiments/fuse_cutout/checkpoints/checkpoint.pt
