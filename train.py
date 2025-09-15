import os
import argparse
import tensorflow as tf

from yolo11_tf.model import build_yolo11
from yolo11_tf.data import build_trainer_dataset
from yolo11_tf.train_utils import Trainer, TrainConfig
from yolo11_tf.metrics import evaluate_dataset_pr_maps


def parse_args():
    ap = argparse.ArgumentParser("Train YOLO11-TF (Ultralytics-style head)")
    ap.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--model_scale", type=str, default="n", choices=["n","s","m","l","x"], help="Model size multiplier")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="runs/train/exp")
    ap.add_argument("--log_every", type=int, default=20, help="Print train metrics every N steps")
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
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            print(f"Using {len(gpus)} GPU(s) with memory growth enabled", flush=True)
        else:
            print("No GPU found, running on CPU", flush=True)
    except Exception as e:
        print(f"GPU setup warning: {e}", flush=True)
    import time
    # Local imports for dataset utilities
    from yolo11_tf.data import load_yolo_yaml, build_file_list

    train_ds, num_classes = build_trainer_dataset(args.data, imgsz=args.imgsz, batch_size=args.batch, split="train", shuffle=True)
    val_ds, _ = build_trainer_dataset(args.data, imgsz=args.imgsz, batch_size=args.batch, split="val", shuffle=False)
    # Steps per epoch for progress bar
    train_dir, _, _ = load_yolo_yaml(args.data)
    train_files = build_file_list(train_dir)
    steps_per_epoch = max(1, len(train_files) // args.batch)

    width_mult, depth_mult = scale_to_multipliers(args.model_scale)
    model = build_yolo11(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult)

    cfg = TrainConfig(num_classes=num_classes, img_size=args.imgsz, reg_max=16, lr=args.lr)
    trainer = Trainer(model, cfg)

    # TensorBoard writer for charts
    tb_dir = os.path.join(args.out, "tb")
    writer = tf.summary.create_file_writer(tb_dir)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        t0 = time.time()
        seen = 0
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
        # Evaluate PR/mAP@0.5 and mAP@0.5:0.95 on val set
        p, r, m50, m5095 = evaluate_dataset_pr_maps(model, val_ds, num_classes, conf_thres=0.05, imgsz=args.imgsz)
        dt = time.time() - t0
        ips = (seen / dt) if dt > 0 else 0.0
        print(
            f"Epoch {epoch+1}/{args.epochs} done in {dt:.1f}s ({ips:.1f} img/s) - "
            f"last_loss={metrics['loss']:.3f}  P={p:.4f} R={r:.4f} mAP50={m50:.4f} mAP50-95={m5095:.4f}",
            flush=True,
        )
        # Log to TensorBoard for charts
        with writer.as_default():
            tf.summary.scalar('metrics/precision', p, step=epoch)
            tf.summary.scalar('metrics/recall', r, step=epoch)
            tf.summary.scalar('metrics/mAP50', m50, step=epoch)
            tf.summary.scalar('metrics/mAP50_95', m5095, step=epoch)
            tf.summary.scalar('loss/train', metrics['loss'], step=epoch)
            writer.flush()


if __name__ == "__main__":
    main()

