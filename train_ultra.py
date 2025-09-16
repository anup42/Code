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
    ap.add_argument("--log_every", type=int, default=20, help="Print train metrics every N steps")
    ap.add_argument("--steps_per_epoch", type=int, default=0, help="Override steps per epoch; if >0, validate after N steps")
    ap.add_argument("--val_steps", type=int, default=0, help="Optional cap on validation steps per epoch (0 = full val)")
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
    # Enable GPU memory growth to avoid full allocation
    try:
        # Print TensorFlow version and build info for easier debugging
        try:
            print(f"TensorFlow version: {tf.__version__}", flush=True)
            bi = tf.sysconfig.get_build_info()
            cuda_v = bi.get('cuda_version', '?')
            cudnn_v = bi.get('cudnn_version', '?')
            print(f"TF build CUDA: {cuda_v} cuDNN: {cudnn_v}", flush=True)
        except Exception:
            pass
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            print(f"Using {len(gpus)} GPU(s) with memory growth enabled: {gpus}", flush=True)
        else:
            print("No GPU found, running on CPU", flush=True)
    except Exception as e:
        print(f"GPU setup warning: {e}", flush=True)
    import time

    # Datasets
    train_ds, num_classes = build_trainer_dataset(args.data, imgsz=args.imgsz, batch_size=args.batch, split="train", shuffle=True)
    val_ds, _ = build_trainer_dataset(args.data, imgsz=args.imgsz, batch_size=args.batch, split="val", shuffle=False)
    # Steps per epoch
    from yolo11_tf.data import load_yolo_yaml, build_file_list
    train_dir, val_dir, _ = load_yolo_yaml(args.data)
    train_files = build_file_list(train_dir)
    val_files = build_file_list(val_dir)
    estimated_steps = max(1, len(train_files) // args.batch)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch and args.steps_per_epoch > 0 else estimated_steps
    if steps_per_epoch > estimated_steps:
        train_ds = train_ds.repeat()
    steps_per_val = max(1, len(val_files) // args.batch)
    if args.val_steps and args.val_steps > 0:
        steps_per_val = args.val_steps

    # Model
    width_mult, depth_mult = scale_to_multipliers(args.model_scale)
    model = build_yolo11(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult, reg_max=16)

    # Trainer
    cfg = TrainConfig(num_classes=num_classes, img_size=args.imgsz, reg_max=16, lr=args.lr)
    trainer = Trainer(model, cfg)

    # Simple training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        t0 = time.time()
        seen = 0
        # Train epoch with progress bar
        pb = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=["loss","cls","box","dfl","pos"], unit_name="batch")
        for step, (images, targets) in enumerate(train_ds, start=1):
            try:
                seen += int(images.shape[0])
            except Exception:
                pass
            metrics = trainer.train_step(images, targets)
            if step <= steps_per_epoch:
                pb.update(step, values=[
                    ("loss", metrics['loss']), ("cls", metrics['cls']), ("box", metrics['box']), ("dfl", metrics['dfl']), ("pos", metrics['pos'])
                ])
            if step >= steps_per_epoch:
                break
        # Eval quick metrics at epoch end (mAP@0.5 on val)
        val_iter = val_ds.take(steps_per_val).map(lambda img, tgt: ((img, tf.zeros([tf.shape(tgt)[0], cfg.max_boxes, 5], dtype=tf.float32)), tgt))
        p, r, m = evaluate_dataset_map50(model, val_iter, num_classes, conf_thres=0.05, iou_thres=0.5, imgsz=args.imgsz)
        dt = time.time() - t0
        ips = (seen / dt) if dt > 0 else 0.0
        print(
            f"Epoch {epoch+1}/{args.epochs} done in {dt:.1f}s ({ips:.1f} img/s) - "
            f"last_loss={metrics['loss']:.3f}  P={p:.4f} R={r:.4f} mAP50={m:.4f}",
            flush=True,
        )
        print(f"Epoch {epoch+1}/{args.epochs} — loss={metrics['loss']:.2f}  P={p:.4f} R={r:.4f} mAP50={m:.4f}")


if __name__ == "__main__":
    main()
