"""
Convert COCO segmentation format to YOLO segmentation format.
Uses Ultralytics built-in converter for reliable conversion.
"""

import os
from pathlib import Path
from ultralytics.data.converter import convert_coco

# Paths
TRAINING_DIR = "/root/indriya/count_training"
COCO_DIR = os.path.join(TRAINING_DIR, "coco_dataset")
YOLO_DIR = os.path.join(TRAINING_DIR, "dataset")

print("=" * 70)
print("Converting COCO Segmentation to YOLO Format")
print("=" * 70)
print(f"Source (COCO): {COCO_DIR}")
print(f"Target (YOLO): {YOLO_DIR}")
print("Using Ultralytics built-in converter")
print("=" * 70)

# Main conversion process
try:
    # Check if COCO dataset exists
    if not os.path.exists(COCO_DIR):
        print(f"\n❌ ERROR: COCO dataset not found at {COCO_DIR}")
        print("Please run download_dataset.py first!")
        exit(1)
    
    # Look for annotation files
    annotation_files = list(Path(COCO_DIR).glob("**/_annotations.coco.json"))
    
    if not annotation_files:
        print(f"\n❌ ERROR: No COCO annotation files found in {COCO_DIR}")
        print("Run download_dataset.py to download the dataset first.")
        exit(1)
    
    print(f"\n✅ Found COCO dataset")
    print(f"   Annotation files: {len(annotation_files)}")
    
    # Create output directory
    os.makedirs(YOLO_DIR, exist_ok=True)
    
    print(f"\n🔄 Converting COCO to YOLO format...")
    print(f"   This may take a few minutes...")
    
    # Use Ultralytics built-in converter
    # It handles all the complexity: normalization, file structure, etc.
    convert_coco(
        labels_dir=COCO_DIR,        # Path to COCO dataset directory
        save_dir=YOLO_DIR,          # Where to save YOLO format
        use_segments=True,          # Convert segmentation (polygons), not just boxes
        use_keypoints=False,        # We don't have keypoints
        cls91to80=False             # Don't remap classes
    )
    
    print("\n" + "=" * 70)
    print("✅ Conversion completed successfully!")
    print("=" * 70)
    print(f"YOLO dataset ready at: {YOLO_DIR}")
    print(f"\nStructure:")
    print(f"  - Images: {YOLO_DIR}/images/")
    print(f"  - Labels: {YOLO_DIR}/labels/")
    print(f"  - Config: {YOLO_DIR}/data.yaml")
    print("\n💡 You can now run train_segmentation.py to start training!")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Conversion failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure COCO dataset is downloaded (run download_dataset.py)")
    print("  2. Check that annotations exist in coco_dataset/")
    print("  3. Verify the dataset has segmentation (polygon) annotations")
    import traceback
    traceback.print_exc()
    exit(1)
