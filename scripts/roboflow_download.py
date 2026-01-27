"""
Roboflow Dataset Downloader
Downloads annotated dataset in COCO format from Roboflow using API
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print("ERROR: roboflow package not installed")
    print("Install with: pip install roboflow")
    sys.exit(1)


def download_dataset(api_key, workspace, project, version, output_dir):
    """
    Download dataset from Roboflow in COCO format

    Args:
        api_key: Your Roboflow API key
        workspace: Your workspace name (usually your username)
        project: Project name (e.g., 'fuse-cutout-detection')
        version: Dataset version number (usually 1 for first version)
        output_dir: Where to save the downloaded dataset
    """
    print("=" * 70)
    print("ROBOFLOW DATASET DOWNLOADER")
    print("=" * 70)

    # Initialize Roboflow
    print(f"\n1. Connecting to Roboflow...")
    try:
        rf = Roboflow(api_key=api_key)
        print("   ✓ Connected successfully")
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        sys.exit(1)

    # Get project
    print(f"\n2. Accessing project: {workspace}/{project}")
    try:
        project_obj = rf.workspace(workspace).project(project)
        print(f"   ✓ Project found")
    except Exception as e:
        print(f"   ✗ Failed to access project: {e}")
        print(f"\nTip: Check your workspace and project names in Roboflow URL:")
        print(f"     https://app.roboflow.com/YOUR_WORKSPACE/YOUR_PROJECT")
        sys.exit(1)

    # Get specific version
    print(f"\n3. Fetching version {version}...")
    try:
        dataset = project_obj.version(version)
        print(f"   ✓ Version {version} found")
    except Exception as e:
        print(f"   ✗ Failed to get version: {e}")
        print(f"\nTip: Make sure you've generated a dataset version in Roboflow")
        print(f"     Go to your project → 'Generate' → Create version")
        sys.exit(1)

    # Download dataset
    print(f"\n4. Downloading dataset in COCO format to {output_dir}...")
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Download in COCO format
        dataset.download(
            model_format="coco",
            location=output_dir
        )
        print(f"   ✓ Download complete!")
    except Exception as e:
        print(f"   ✗ Download failed: {e}")
        sys.exit(1)

    # Verify downloaded structure
    print(f"\n5. Verifying downloaded files...")
    dataset_path = Path(output_dir)

    # Roboflow typically creates a subdirectory with project name
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if subdirs:
        actual_path = subdirs[0]
        print(f"   ✓ Dataset extracted to: {actual_path}")

        # Check for expected directories
        train_dir = actual_path / "train"
        valid_dir = actual_path / "valid"
        test_dir = actual_path / "test"

        if train_dir.exists():
            train_images = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
            print(f"   ✓ Train: {train_images} images")

        if valid_dir.exists():
            valid_images = len(list(valid_dir.glob("*.jpg"))) + len(list(valid_dir.glob("*.png")))
            print(f"   ✓ Valid: {valid_images} images")

        if test_dir.exists():
            test_images = len(list(test_dir.glob("*.jpg"))) + len(list(test_dir.glob("*.png")))
            print(f"   ✓ Test: {test_images} images")

        # Check for COCO annotation files
        json_files = list(actual_path.rglob("*.json"))
        if json_files:
            print(f"   ✓ Found {len(json_files)} annotation files:")
            for json_file in json_files:
                print(f"     - {json_file.name}")

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Validate annotations: python scripts/roboflow_validator.py")
    print(f"2. Visualize samples: python scripts/visualize_annotations.py")
    print(f"3. Start training: python scripts/train.py")
    print()


def interactive_mode():
    """Interactive mode to gather information from user"""
    print("=" * 70)
    print("ROBOFLOW DATASET DOWNLOADER - INTERACTIVE MODE")
    print("=" * 70)
    print()
    print("You'll need the following information from Roboflow:")
    print("1. API Key (Settings → API Key)")
    print("2. Workspace name (in URL or dashboard)")
    print("3. Project name (in URL or dashboard)")
    print("4. Version number (usually 1 for first version)")
    print()

    api_key = input("Enter your Roboflow API key: ").strip()
    workspace = input("Enter workspace name: ").strip()
    project = input("Enter project name: ").strip()
    version = input("Enter version number (default: 1): ").strip() or "1"
    output_dir = input("Enter output directory (default: sam3_fuse_cutout_dataset): ").strip() or "sam3_fuse_cutout_dataset"

    print()
    return api_key, workspace, project, int(version), output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download annotated dataset from Roboflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for first time)
  python scripts/roboflow_download.py

  # Command line mode
  python scripts/roboflow_download.py \\
    --api-key YOUR_API_KEY \\
    --workspace your-workspace \\
    --project fuse-cutout-detection \\
    --version 1 \\
    --output sam3_fuse_cutout_dataset

Finding your information:
  1. Go to https://app.roboflow.com
  2. Open your project
  3. URL format: https://app.roboflow.com/WORKSPACE/PROJECT/VERSION
  4. API key: Click profile → Settings → Copy API Key
        """
    )

    parser.add_argument("--api-key", help="Roboflow API key")
    parser.add_argument("--workspace", help="Workspace name")
    parser.add_argument("--project", help="Project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version (default: 1)")
    parser.add_argument("--output", default="sam3_fuse_cutout_dataset", help="Output directory")

    args = parser.parse_args()

    # If no arguments provided, use interactive mode
    if not args.api_key or not args.workspace or not args.project:
        api_key, workspace, project, version, output_dir = interactive_mode()
    else:
        api_key = args.api_key
        workspace = args.workspace
        project = args.project
        version = args.version
        output_dir = args.output

    # Download dataset
    download_dataset(api_key, workspace, project, version, output_dir)


if __name__ == "__main__":
    main()
