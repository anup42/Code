import os
import argparse
import tensorflow as tf

from yolo11_tf.model import build_yolo11
from yolo11_tf.data import build_trainer_dataset
from yolo11_tf.train_utils import Trainer, TrainConfig
from yolo11_tf.metrics import evaluate_dataset_map50


def parse_args():
    ap = argparse.ArgumentParser("Train YOLO11-TF (Ultralytics-style head) with DFL")
    ap.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--model_scale", type=str, default="n", choices=["n","s","m","l","x"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="runs/train_ultra/exp")
    return ap.parse_args()


def scale_to_multipliers(scale: str):
    table = {
        "n": (0.50, 0.33),
        "s": (0.75, 0.50),
        "m": (1.00, 0.67),
        "l": (1.25, 1.00),
        "x": (1.50, 1.00),
    }
    return table.get(scale, (0.50, 0.33))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Datasets
    train_ds, num_classes = build_trainer_dataset(args.data, imgsz=args.imgsz, batch_size=args.batch, split="train", shuffle=True)
    val_ds, _ = build_trainer_dataset(args.data, imgsz=args.imgsz, batch_size=args.batch, split="val", shuffle=False)

    # Model
    width_mult, depth_mult = scale_to_multipliers(args.model_scale)
    model = build_yolo11(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult, reg_max=16)

    # Trainer
    cfg = TrainConfig(num_classes=num_classes, img_size=args.imgsz, reg_max=16, lr=args.lr)
    trainer = Trainer(model, cfg)

    # Simple training loop
    for epoch in range(args.epochs):
        # Train epoch
        for images, targets in train_ds:
            metrics = trainer.train_step(images, targets)
        # Eval quick metrics at epoch end (mAP@0.5 on val)
        p, r, m = evaluate_dataset_map50(model, val_ds.map(lambda img, tgt: ((img, tf.zeros([tf.shape(tgt)[0], cfg.max_boxes, 5], dtype=tf.float32)), tgt)), num_classes, conf_thres=0.05, iou_thres=0.5)
        print(f"Epoch {epoch+1}/{args.epochs} — loss={metrics['loss']:.2f}  P={p:.4f} R={r:.4f} mAP50={m:.4f}")


if __name__ == "__main__":
    main()

