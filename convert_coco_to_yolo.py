"""
Convert COCO segmentation format to YOLO segmentation format.
Custom implementation for Roboflow dataset structure.
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image

# Paths - use current script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
COCO_DIR = os.path.join(DATA_DIR, "coco_dataset")
YOLO_DIR = os.path.join(DATA_DIR, "dataset")

print("=" * 70)
print("Converting COCO Segmentation to YOLO Format")
print("=" * 70)
print(f"Source (COCO): {COCO_DIR}")
print(f"Target (YOLO): {YOLO_DIR}")
print("Using custom converter for Roboflow dataset")
print("=" * 70)


def convert_polygon_to_yolo(polygon, img_width, img_height):
    """Convert COCO polygon format to YOLO segmentation format (normalized)."""
    # COCO polygon is [x1, y1, x2, y2, ..., xn, yn]
    # YOLO needs [x1/w, y1/h, x2/w, y2/h, ..., xn/w, yn/h] (all normalized)
    normalized_points = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / img_width
        y = polygon[i + 1] / img_height
        normalized_points.extend([x, y])
    return normalized_points


def convert_coco_split(coco_split_dir, yolo_split_dir, split_name):
    """Convert one split (train/valid/test) from COCO to YOLO format."""
    print(f"\n📂 Processing {split_name} split...")
    
    # Paths
    annotation_file = os.path.join(coco_split_dir, "_annotations.coco.json")
    images_output_dir = os.path.join(yolo_split_dir, "images", split_name)
    labels_output_dir = os.path.join(yolo_split_dir, "labels", split_name)
    
    # Create output directories
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # Load COCO annotations
    if not os.path.exists(annotation_file):
        print(f"   ⚠️  No annotations found, skipping {split_name}")
        return None
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build category mapping (COCO category_id to YOLO class index)
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    category_names = [cat['name'] for cat in coco_data['categories']]
    
    # Build image id to filename mapping
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    converted_count = 0
    
    # Process each image
    for img_id, img_info in images_dict.items():
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Copy image file
        src_img_path = os.path.join(coco_split_dir, img_filename)
        dst_img_path = os.path.join(images_output_dir, img_filename)
        
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"   ⚠️  Image not found: {src_img_path}")
            continue
        
        # Create YOLO label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(labels_output_dir, label_filename)
        
        with open(label_path, 'w') as f:
            # Get annotations for this image
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    # Get class index
                    class_idx = categories[ann['category_id']]
                    
                    # Convert segmentation polygons
                    if 'segmentation' in ann and ann['segmentation']:
                        for polygon in ann['segmentation']:
                            if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                                yolo_polygon = convert_polygon_to_yolo(polygon, img_width, img_height)
                                # Write: class_idx x1 y1 x2 y2 ... xn yn
                                f.write(f"{class_idx} " + " ".join(f"{coord:.6f}" for coord in yolo_polygon) + "\n")
        
        converted_count += 1
    
    print(f"   ✅ Converted {converted_count} images")
    return category_names


# Main conversion process
try:
    # Check if COCO dataset exists
    if not os.path.exists(COCO_DIR):
        print(f"\n❌ ERROR: COCO dataset not found at {COCO_DIR}")
        print("Please run download_dataset.py first!")
        exit(1)
    
    # Clean output directory if it exists
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)
    os.makedirs(YOLO_DIR, exist_ok=True)
    
    print(f"\n🔄 Starting conversion...")
    
    # Convert each split
    splits = ['train', 'valid', 'test']
    class_names = None
    
    for split in splits:
        split_dir = os.path.join(COCO_DIR, split)
        if os.path.exists(split_dir):
            result = convert_coco_split(split_dir, YOLO_DIR, split)
            if result and class_names is None:
                class_names = result
    
    # Create data.yaml file
    if class_names:
        data_yaml_path = os.path.join(YOLO_DIR, "data.yaml")
        with open(data_yaml_path, 'w') as f:
            f.write(f"# YOLO Dataset Configuration\n")
            f.write(f"# Auto-generated from Roboflow COCO dataset\n\n")
            f.write(f"path: {YOLO_DIR}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/valid\n")
            if os.path.exists(os.path.join(COCO_DIR, 'test')):
                f.write(f"test: images/test\n")
            f.write(f"\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names: {class_names}\n")
        
        print(f"\n✅ Created data.yaml with {len(class_names)} classes: {class_names}")
    
    print("\n" + "=" * 70)
    print("✅ Conversion completed successfully!")
    print("=" * 70)
    print(f"YOLO dataset ready at: {YOLO_DIR}")
    print(f"\nStructure:")
    print(f"  - Images: {YOLO_DIR}/images/{{train,valid,test}}/")
    print(f"  - Labels: {YOLO_DIR}/labels/{{train,valid,test}}/")
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
