# YOLO Instance Segmentation Training for Jewelry Detection

This folder contains scripts and resources for training a YOLO instance segmentation model on the jewelry dataset from Roboflow.

## 📁 Folder Structure

```
counting_jewelry/
├── download_dataset.py         # Download dataset from Roboflow (COCO format)
├── convert_coco_to_yolo.py     # Convert COCO to YOLO segmentation format
├── train_segmentation.py       # Train YOLO instance segmentation model
├── test_model.py               # Test trained models
├── run_training.sh             # Automated training pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── data/                       # All data stored here (gitignored except important files)
    ├── coco_dataset/           # COCO format dataset (created after download)
    ├── dataset/                # YOLO format dataset (created after conversion)
    │   └── data.yaml           # YOLO config (KEPT in git)
    ├── runs/                   # Training runs and checkpoints
    │   └── train/weights/      # All epoch weights (KEPT in git)
    ├── models/                 # Timestamped model backups (KEPT in git)
    └── venv/                   # Virtual environment (gitignored)
```

## 🚀 Quick Start

### Option 1: Automated Pipeline (Recommended)

```bash
cd /root/counting_jewelry
./run_training.sh
```

This will automatically:
1. Create virtual environment
2. Install dependencies
3. Download COCO dataset from Roboflow
4. Convert COCO to YOLO format
5. Start training

### Option 2: Manual Steps

**1. Install Dependencies**

```bash
cd /root/counting_jewelry
pip install -r requirements.txt
```

**2. Download Dataset (COCO format)**

```bash
python download_dataset.py
```

**3. Convert COCO to YOLO Format**

```bash
python convert_coco_to_yolo.py
```

**4. Train the Model**

```bash
python train_segmentation.py
```

## ⚙️ Training Configuration

The training script uses the following settings:

- **Model**: YOLO26-Large Segmentation (`yolo26l-seg.pt`)
- **Epochs**: 80
- **Image Size**: 960x960
- **Batch Size**: 4
- **Device**: GPU (0)
- **Workers**: 4
- **Optimizer**: AdamW
- **Learning Rate**: 0.001 (with cosine decay)
- **Save Period**: Every epoch
- **Early Stopping**: Patience of 10 epochs

You can modify these settings by editing [train_segmentation.py](train_segmentation.py).

## 📊 Output Files

After training, you'll find:

1. **Training runs**: `data/runs/train/`
   - Weights: `data/runs/train/weights/best.pt` and `data/runs/train/weights/last.pt`
   - Plus all epoch weights: `epoch1.pt`, `epoch2.pt`, ..., `epoch80.pt`
   - Training plots and metrics: `results.csv`, `results.png`, etc.
   - Validation results

2. **Backup models**: `data/models/`
   - Timestamped copies of best and last weights
   - Format: `jewelry_seg_best_YYYYMMDD_HHMMSS.pt`

## 🔄 Resume Training

If training is interrupted, simply run the training script again:

```bash
python train_segmentation.py
```

The script will automatically detect and resume from the last checkpoint.

## 💡 Tips

- Monitor GPU usage with `nvidia-smi`
- Check training progress in real-time: `tail -f data/runs/train/results.csv`
- Adjust batch size if you encounter out-of-memory errors
- Lower image size (e.g., 416) for faster training on limited resources

## 📝 Notes

- This is **instance segmentation** (pixel-level masks, not just bounding boxes)
- Dataset is downloaded in COCO format and automatically converted to YOLO format
- The model outputs both bounding boxes and pixel-level polygon masks
- Conversion preserves all polygon annotations from COCO format
- This training setup is isolated and won't affect the main backend/frontend code

## 🔄 Data Flow

1. **Roboflow** → COCO Segmentation format (JSON annotations + images)
2. **Conversion** → YOLO Segmentation format (txt files with normalized polygons)
3. **Training** → YOLO26 Instance Segmentation model
4. **Output** → Model that predicts polygon masks for jewelry items

## 🐛 Troubleshooting

**COCO dataset not found**: Run `download_dataset.py` first

**YOLO dataset not found**: Run `convert_coco_to_yolo.py` after downloading

**Out of memory**: Reduce `BATCH_SIZE` in `train_segmentation.py` (try 2 or 1)

**No GPU**: Change `DEVICE = 0` to `DEVICE = 'cpu'` in `train_segmentation.py`

**Conversion errors**: Check that COCO annotations exist in `data/coco_dataset/train/_annotations.coco.json`
