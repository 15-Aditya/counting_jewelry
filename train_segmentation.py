"""
Train YOLO Instance Segmentation model on jewelry dataset.
This script supports both fresh training and resuming from checkpoints.
"""

import torch
import os
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# Set up paths - use current script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
PROJECT_DIR = os.path.join(DATA_DIR, "runs")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Training configuration
EPOCHS = 80
IMG_SIZE = 960
BATCH_SIZE = 4
DEVICE = 0  # GPU 0, use 'cpu' if no GPU available

# Create necessary directories
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("YOLO Instance Segmentation Training for Jewelry Detection")
print("=" * 70)
print(f"Script Directory: {SCRIPT_DIR}")
print(f"Data Directory: {DATA_DIR}")
print(f"Dataset Path: {DATA_YAML}")
print(f"Output Directory: {PROJECT_DIR}")
print(f"Models Directory: {MODELS_DIR}")
print(f"Device: {'GPU' if DEVICE == 0 else 'CPU'}")
print("=" * 70)

# Check if dataset exists
if not os.path.exists(DATA_YAML):
    print("\n❌ ERROR: Dataset not found!")
    print(f"Please run download_dataset.py first to download the dataset.")
    print(f"Expected location: {DATA_YAML}")
    exit(1)

# Check for previous training run
train_weights_path = os.path.join(PROJECT_DIR, "train/weights/last.pt")
if os.path.exists(train_weights_path):
    print("\n🔄 Resuming previous training...")
    print(f"Loading checkpoint from: {train_weights_path}")
    model = YOLO(train_weights_path)
    resume_flag = True
else:
    print("\n🆕 Starting fresh training...")
    # Use YOLO26 Large Segmentation model for instance segmentation
    base_model = "yolo26l-seg.pt"
    
    model = YOLO(base_model)
    resume_flag = False


print("\n" + "=" * 70)
print("Starting Training...")
print("=" * 70)
print(f"Epochs: {EPOCHS}")
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Save Period: Every epoch")
print("=" * 70 + "\n")

# Train the model
try:
    results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=4,
    project=PROJECT_DIR,
    name="train",
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    close_mosaic=10,
    save=True,
    save_period=1,
    resume=resume_flag,
    verbose=True,
    patience=10,
    plots=True,
    )
    
    print("\n" + "=" * 70)
    print("✅ Training finished successfully!")
    print("=" * 70)
    
    # Copy best and last weights to models directory
    best_weights = os.path.join(PROJECT_DIR, "train/weights/best.pt")
    last_weights = os.path.join(PROJECT_DIR, "train/weights/last.pt")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.exists(best_weights):
        best_dest = os.path.join(MODELS_DIR, f"jewelry_seg_best_{timestamp}.pt")
        import shutil
        shutil.copy2(best_weights, best_dest)
        print(f"✅ Best model saved to: {best_dest}")
    
    if os.path.exists(last_weights):
        last_dest = os.path.join(MODELS_DIR, f"jewelry_seg_last_{timestamp}.pt")
        import shutil
        shutil.copy2(last_weights, last_dest)
        print(f"✅ Last model saved to: {last_dest}")
    
    print("\n📊 Training Results:")
    print(f"   - Training runs: {PROJECT_DIR}/train/")
    print(f"   - Weights: {PROJECT_DIR}/train/weights/")
    print(f"   - Plots: {PROJECT_DIR}/train/")
    print(f"   - Model backups: {MODELS_DIR}/")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Training failed with error: {e}")
    raise

print("\n✨ All done!")
