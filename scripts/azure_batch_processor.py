"""
Production Batch Processor for Azure Blob Storage
Monitors Azure Blob Storage for new images and processes them with SAM3

Usage:
    python scripts/azure_batch_processor.py \
        --storage-account myaccount \
        --container-name input-images \
        --output-container output-images \
        --checkpoint experiments/fuse_neutrals/checkpoints/checkpoint_epoch_50.pth
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import os

try:
    from azure.storage.blob import BlobServiceClient, BlobClient
    from azure.core.exceptions import ResourceNotFoundError
    import torch
    from PIL import Image
    import io
except ImportError as e:
    print(f"ERROR: Required package not installed: {e}")
    print("Install with: pip install azure-storage-blob torch pillow")
    sys.exit(1)

# Add SAM3 to path
sam3_path = Path(__file__).parent.parent / "sam3"
if sam3_path.exists():
    sys.path.insert(0, str(sam3_path))

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"ERROR: Cannot import SAM3: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AzureBatchProcessor:
    """Process images from Azure Blob Storage using SAM3 model."""

    def __init__(
        self,
        storage_account: str,
        storage_key: str,
        input_container: str,
        output_container: str,
        checkpoint_path: str,
        text_prompt: str = "fuse cutout",
        confidence_threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.input_container = input_container
        self.output_container = output_container
        self.text_prompt = text_prompt
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Initialize Azure Blob client
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account};AccountKey={storage_key};EndpointSuffix=core.windows.net"
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        logger.info(f"Connected to Azure Storage: {storage_account}")

        # Load SAM3 model
        logger.info(f"Loading model from: {checkpoint_path}")
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=device,
            eval_mode=True,
            load_from_HF=False,
        )

        # Create processor
        self.processor = Sam3Processor(
            model=self.model,
            resolution=1008,
            device=device,
            confidence_threshold=confidence_threshold,
        )

        logger.info("Model loaded successfully")

    def list_unprocessed_blobs(self):
        """List all images in input container that haven't been processed."""
        input_client = self.blob_service_client.get_container_client(
            self.input_container
        )
        output_client = self.blob_service_client.get_container_client(
            self.output_container
        )

        # Get all input blobs
        input_blobs = []
        for blob in input_client.list_blobs():
            if blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
                input_blobs.append(blob.name)

        # Get processed blobs (check for corresponding result files)
        processed_blobs = set()
        for blob in output_client.list_blobs():
            # Extract original filename from result filename
            # e.g., "2026-02-12/image_001_result.jpg" -> "image_001.jpg"
            if "_result." in blob.name:
                original_name = blob.name.split("/")[-1].replace("_result.", ".")
                original_name = (
                    original_name.replace(".jpg", "")
                    .replace(".jpeg", "")
                    .replace(".png", "")
                )
                processed_blobs.add(original_name)

        # Find unprocessed
        unprocessed = []
        for blob_name in input_blobs:
            base_name = Path(blob_name).stem
            if base_name not in processed_blobs:
                unprocessed.append(blob_name)

        logger.info(
            f"Found {len(input_blobs)} input images, {len(unprocessed)} unprocessed"
        )
        return unprocessed

    def download_blob_to_image(self, blob_name):
        """Download blob and convert to PIL Image."""
        container_client = self.blob_service_client.get_container_client(
            self.input_container
        )
        blob_client = container_client.get_blob_client(blob_name)

        blob_data = blob_client.download_blob().readall()
        image = Image.open(io.BytesIO(blob_data)).convert("RGB")

        return image

    def run_inference(self, image):
        """Run SAM3 inference on image."""
        with torch.no_grad():
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(prompt=self.text_prompt, state=state)

        boxes = (
            state["boxes"].cpu().numpy()
            if isinstance(state["boxes"], torch.Tensor)
            else state["boxes"]
        )
        masks = (
            state["masks"].cpu().numpy()
            if isinstance(state["masks"], torch.Tensor)
            else state["masks"]
        )
        scores = (
            state["scores"].cpu().numpy()
            if isinstance(state["scores"], torch.Tensor)
            else state["scores"]
        )

        return {
            "boxes": boxes,
            "masks": masks,
            "scores": scores,
            "num_detections": len(boxes),
        }

    def visualize_and_save(self, image, results, output_blob_name):
        """Create visualization and upload to Azure Blob Storage."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(np.array(image))

        # Draw detections
        for box, score in zip(results["boxes"], results["scores"]):
            x0, y0, x1, y1 = box
            width = x1 - x0
            height = y1 - y0

            rect = patches.Rectangle(
                (x0, y0), width, height, linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

            label = f"{score:.2f}"
            ax.text(
                x0,
                y0 - 5,
                label,
                color="lime",
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2),
            )

        ax.set_title(
            f"SAM3: '{self.text_prompt}' | {results['num_detections']} detections",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="jpg", bbox_inches="tight", dpi=150)
        buf.seek(0)
        plt.close()

        # Upload to blob storage
        container_client = self.blob_service_client.get_container_client(
            self.output_container
        )
        blob_client = container_client.get_blob_client(output_blob_name)
        blob_client.upload_blob(buf, overwrite=True)

        logger.info(f"Uploaded result to: {output_blob_name}")

    def save_metadata_json(self, blob_name, results, output_blob_name):
        """Save detection metadata as JSON."""
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_image": blob_name,
            "output_image": output_blob_name,
            "model_prompt": self.text_prompt,
            "confidence_threshold": self.confidence_threshold,
            "num_detections": int(results["num_detections"]),
            "detections": [
                {"box": box.tolist(), "confidence": float(score)}
                for box, score in zip(results["boxes"], results["scores"])
            ],
        }

        json_data = json.dumps(metadata, indent=2)
        json_blob_name = (
            output_blob_name.replace(".jpg", ".json")
            .replace(".jpeg", ".json")
            .replace(".png", ".json")
        )

        container_client = self.blob_service_client.get_container_client(
            self.output_container
        )
        blob_client = container_client.get_blob_client(json_blob_name)
        blob_client.upload_blob(json_data, overwrite=True)

    def process_batch(self, batch_size=50):
        """Process a batch of unprocessed images."""
        unprocessed = self.list_unprocessed_blobs()

        if not unprocessed:
            logger.info("No unprocessed images found")
            return 0

        # Process up to batch_size images
        to_process = unprocessed[:batch_size]
        logger.info(f"Processing batch of {len(to_process)} images")

        results_summary = []
        today = datetime.utcnow().strftime("%Y-%m-%d")

        for i, blob_name in enumerate(to_process, 1):
            try:
                logger.info(f"[{i}/{len(to_process)}] Processing: {blob_name}")

                # Download and process
                image = self.download_blob_to_image(blob_name)
                results = self.run_inference(image)

                logger.info(f"  ✓ Found {results['num_detections']} detections")

                # Create output blob name with date prefix
                base_name = Path(blob_name).stem
                output_blob_name = f"{today}/{base_name}_result.jpg"

                # Visualize and upload
                self.visualize_and_save(image, results, output_blob_name)

                # Save metadata
                self.save_metadata_json(blob_name, results, output_blob_name)

                results_summary.append(
                    {
                        "input": blob_name,
                        "output": output_blob_name,
                        "detections": results["num_detections"],
                    }
                )

            except Exception as e:
                logger.error(f"  ✗ Error processing {blob_name}: {e}", exc_info=True)
                continue

        logger.info(
            f"✓ Batch processing complete: {len(results_summary)}/{len(to_process)} successful"
        )
        return len(results_summary)


def main():
    parser = argparse.ArgumentParser(description="Azure Batch Processor for SAM3")

    parser.add_argument(
        "--storage-account", required=True, help="Azure Storage account name"
    )
    parser.add_argument(
        "--storage-key", required=True, help="Azure Storage account key"
    )
    parser.add_argument(
        "--input-container", required=True, help="Input blob container name"
    )
    parser.add_argument(
        "--output-container", required=True, help="Output blob container name"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--text-prompt", default="fuse cutout", help="Text prompt for detection"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Number of images per batch"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device to use"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously (poll for new images)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="Poll interval in seconds (for continuous mode)",
    )

    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Initialize processor
    processor = AzureBatchProcessor(
        storage_account=args.storage_account,
        storage_key=args.storage_key,
        input_container=args.input_container,
        output_container=args.output_container,
        checkpoint_path=args.checkpoint,
        text_prompt=args.text_prompt,
        confidence_threshold=args.threshold,
        device=args.device,
    )

    if args.continuous:
        logger.info(
            f"Running in continuous mode (poll interval: {args.poll_interval}s)"
        )
        import time

        while True:
            try:
                processed = processor.process_batch(batch_size=args.batch_size)
                if processed == 0:
                    logger.info(f"No new images, sleeping for {args.poll_interval}s...")
                time.sleep(args.poll_interval)
            except KeyboardInterrupt:
                logger.info("Stopping continuous processing")
                break
            except Exception as e:
                logger.error(f"Error in continuous processing: {e}", exc_info=True)
                time.sleep(args.poll_interval)
    else:
        # Single batch
        processor.process_batch(batch_size=args.batch_size)
        logger.info("Single batch processing complete")


if __name__ == "__main__":
    main()
