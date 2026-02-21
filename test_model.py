"""
Test trained YOLO instance segmentation model on sample images.
"""

import os
from ultralytics import YOLO
from pathlib import Path

# Set up paths - use current script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
TEST_DIR = os.path.join(DATA_DIR, "test_results")

# Create test results directory
os.makedirs(TEST_DIR, exist_ok=True)

print("=" * 70)
print("YOLO Instance Segmentation Model Testing")
print("=" * 70)

# Find the latest best model
model_files = list(Path(MODELS_DIR).glob("jewelry_seg_best_*.pt"))
if not model_files:
    # Try the training directory
    best_weight = os.path.join(DATA_DIR, "runs/train/weights/best.pt")
    if os.path.exists(best_weight):
        model_path = best_weight
        print(f"Using model from training directory: {model_path}")
    else:
        print("\n❌ No trained model found!")
        print("Please train a model first using train_segmentation.py")
        exit(1)
else:
    # Use the latest model
    model_path = str(max(model_files, key=os.path.getctime))
    print(f"Using latest model: {model_path}")

# Load model
print("\n📦 Loading model...")
model = YOLO(model_path)

print("\n" + "=" * 70)
print("Model Information:")
print("=" * 70)
print(f"Model: {model_path}")
print(f"Task: Instance Segmentation")
print(f"Classes: {model.names}")
print("=" * 70)

# You can test on specific images
test_image_path = input("\nEnter image path to test (or press Enter to skip): ").strip()

if test_image_path and os.path.exists(test_image_path):
    print(f"\n🔍 Running inference on: {test_image_path}")
    
    results = model.predict(
        source=test_image_path,
        save=True,
        project=TEST_DIR,
        name="predictions",
        conf=0.25,
        iou=0.7,
    )
    
    # Print results
    for result in results:
        print(f"\n📊 Detection Results:")
        print(f"   - Number of objects detected: {len(result.boxes)}")
        if result.masks is not None:
            print(f"   - Segmentation masks: {len(result.masks)}")
        print(f"   - Output saved to: {result.save_dir}")
    
    print("\n✅ Inference complete!")
else:
    print("\n💡 To test the model on images, run:")
    print(f"   python test_model.py")
    print("\n   Then provide an image path when prompted.")
    print("\n   Or use the model programmatically:")
    print(f"   model = YOLO('{model_path}')")
    print("   results = model.predict('your_image.jpg')")

print("\n" + "=" * 70)
