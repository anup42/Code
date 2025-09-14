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
- `pip install tensorflow pillow numpy`
  - GPU: install a matching TensorFlow build per your CUDA/CUDNN.

Train
- Example (COCO-like 80 classes):
  - `python train.py --data C:/path/to/dataset --classes 80 --epochs 100 --imgsz 640 --batch 8`
- Weights are saved to `runs/train/exp/` as `weights_epoch*.h5` and `best.h5`.

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

