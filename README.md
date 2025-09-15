TensorFlow YOLO11-Style Object Detection

Overview
- Anchor-free, decoupled-head TensorFlow/Keras implementation inspired by Ultralytics YOLO11.
- Uses YOLO dataset format (images/ and labels/ with normalized `class x y w h`).
- Includes training and inference pipelines compatible with square input sizes (e.g., 640).

Project Layout
- `yolo11_tf/model.py`: YOLO11-style backbone+neck and decoupled DFL head.
- `yolo11_tf/layers.py`: Core building blocks (ConvBNAct, C2f, SPPF, UpSample).
- `yolo11_tf/losses.py`: DFL, CIoU, BCE and integral distribution utilities.
- `yolo11_tf/data.py`: YOLO-format data loader with letterbox and basic augs.
- `yolo11_tf/train_utils.py`: Simplified one-to-one target assignment and trainer.
- `yolo11_tf/inference.py`: NMS-wrapped inference for predictions.
- `train.py`: CLI to train the model.
- `test.py`: CLI to run inference on images or a folder.

Dataset Format (Ultralytics/YOLO)
- Folder structure:
  - `dataset/images/train/*.jpg|png`, `dataset/labels/train/*.txt`
  - `dataset/images/val/*`, `dataset/labels/val/*`
- Label file lines: `class x_center y_center width height` (normalized 0–1).

Install
- Python 3.9+
- `pip install tensorflow pillow numpy pyyaml`
  - GPU: install a matching TensorFlow build per your CUDA/CUDNN.

Train
- Example (COCO-like 80 classes):
  - `python train.py --data C:/path/to/dataset --classes 80 --epochs 100 --imgsz 640 --batch 8`
- Weights are saved to `runs/train/exp/` as `weights_epoch*.h5` and `best.h5`.

COCO2017
- Prepare dataset (downloads images and annotations, converts to YOLO labels, writes YAML):
  - `python scripts/prepare_coco.py --out datasets/coco2017`
- Train on COCO2017 (80 classes):
  - `python train.py --data datasets/coco2017.yaml --imgsz 640 --batch 64 --epochs 100 --model_scale s --out runs/train/coco2017`
  - Adjust `--batch` and `--epochs` per your hardware.

COCO128 Quickstart
- Download the tiny COCO128 dataset via Ultralytics (optional, for a fast smoke test):
  - `pip install ultralytics`
  - `yolo download dataset=coco128`
- Use the Ultralytics YAML directly for a quick train/eval run (80 classes):
  - On most installs, you can reference the YAML at:
    - `python - <<"PY"
import inspect, os, ultralytics
print(os.path.join(os.path.dirname(inspect.getfile(ultralytics)), 'cfg', 'datasets', 'coco128.yaml'))
PY`
  - Then train 1 epoch for a quick sanity check (replace <path-to-coco128.yaml>):
    - `python train.py --data <path-to-coco128.yaml> --imgsz 640 --batch 16 --epochs 1 --model_scale n --out runs/train/coco128`
- Optional inference on COCO128 val images after training:
  - `python test.py --weights runs/train/coco128/best.h5 --classes 80 --imgsz 640 --source datasets/coco128/images/val --conf 0.25 --iou 0.7`

Inference
- Single image or folder:
  - `python test.py --weights runs/train/exp/best.h5 --classes 80 --imgsz 640 --source C:/path/to/images --conf 0.25 --iou 0.7`
- Outputs boxes, scores and classes to stdout.

Notes and Parity
- Architecture mirrors YOLO11 design elements (C2f, SPPF, PAN-FPN, decoupled head with DFL).
- Training uses a simplified, deterministic one-to-one target assignment (closest point per GT, scale by size). Ultralytics uses task-aligned/dynamic assignment; this simplified approach trades some accuracy for clarity and TensorFlow-compatibility.
- Data input format is exactly Ultralytics YOLO (normalized `.txt` labels). Images are letterboxed to `imgsz`.
- Losses: BCE for classification, DFL + CIoU for box regression.

Customization
- Change model width/depth multipliers via `build_yolo11(width_mult, depth_mult)` for different capacities.
- Adjust `reg_max` and loss weights in `TrainConfig`.
- Extend data augmentations in `yolo11_tf/data.py`.

Disclaimer
- This is a faithful TensorFlow port of the YOLO11 family concepts, but not an official Ultralytics implementation. Exact training dynamics, hyperparameters, and assignment heuristics differ, which may lead to different accuracy/speed tradeoffs.
