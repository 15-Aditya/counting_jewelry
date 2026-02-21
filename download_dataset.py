"""
Download jewelry dataset from Roboflow in COCO segmentation format.
"""

from roboflow import Roboflow
import os
from pathlib import Path

# Set up paths - use current script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
COCO_DIR = os.path.join(DATA_DIR, "coco_dataset")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("Downloading Jewelry Dataset (COCO Instance Segmentation)")
print("=" * 60)

# Initialize Roboflow
rf = Roboflow(api_key="j54VJ7AgTXw4STfaV1ZS")
project = rf.workspace("jewellerysegmentation").project("jewelry-segmentation-fy470")
version = project.version(5)

# Download dataset in COCO segmentation format
print(f"\nDownloading COCO dataset to: {COCO_DIR}")
dataset = version.download("coco-segmentation", location=COCO_DIR)

print("\n" + "=" * 60)
print("COCO dataset downloaded successfully!")
print(f"Dataset location: {COCO_DIR}")
print("\nNext step: Run convert_coco_to_yolo.py to convert to YOLO format")
print("=" * 60)
