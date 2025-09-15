import os
import argparse
import tensorflow as tf

from yolo11_tf.model import build_yolo11
from yolo11_tf.data import build_trainer_dataset
from yolo11_tf.train_utils import Trainer, TrainConfig
from yolo11_tf.metrics import evaluate_dataset_pr_maps, evaluate_dataset_pr_maps_fast


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
    ap.add_argument("--steps_per_epoch", type=int, default=0, help="Override steps per epoch; if >0, run validation after N steps and increment epoch")
    ap.add_argument("--val_steps", type=int, default=0, help="Optional cap on validation steps per epoch (0 = use full val set)")
    ap.add_argument("--cache", action="store_true", help="Cache training dataset in memory for speed (small datasets)")
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
    # Steps per epoch
    train_dir, val_dir, _ = load_yolo_yaml(args.data)
    train_files = build_file_list(train_dir)
    val_files = build_file_list(val_dir)
    estimated_steps = max(1, len(train_files) // args.batch)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch and args.steps_per_epoch > 0 else estimated_steps
    # If user requests more steps than available batches, repeat the dataset to supply enough batches
    if steps_per_epoch > estimated_steps:
        train_ds = train_ds.repeat()
    # Validation steps
    steps_per_val = max(1, len(val_files) // args.batch)
    if args.val_steps and args.val_steps > 0:
        steps_per_val = args.val_steps
    # Optional cache and non-deterministic order for throughput
    if hasattr(args, 'cache') and getattr(args, 'cache'):
        train_ds = train_ds.cache()
    opt = tf.data.Options()
    opt.deterministic = False
    train_ds = train_ds.with_options(opt)

    width_mult, depth_mult = scale_to_multipliers(args.model_scale)
    model = build_yolo11(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult)

    cfg = TrainConfig(num_classes=num_classes, img_size=args.imgsz, reg_max=16, lr=args.lr)
    trainer = Trainer(model, cfg)

    # TensorBoard writer for charts (kept)
    tb_dir = os.path.join(args.out, "tb")
    writer = tf.summary.create_file_writer(tb_dir)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        t0 = time.time()
        seen = 0
        for step, (images, targets) in enumerate(train_ds, start=1):
            try:
                seen += int(images.shape[0])
            except Exception:
                pass
            metrics = trainer.train_step(images, targets)
            # Periodic logging instead of progress bar
            if (step % max(1, int(args.log_every)) == 0) or step == 1:
                print(
                    f"step {step}/{steps_per_epoch} "
                    f"loss={float(metrics['loss']):.3f} cls={float(metrics['cls']):.3f} box={float(metrics['box']):.3f} "
                    f"dfl={float(metrics['dfl']):.3f} obj={float(metrics.get('obj', 0.0)):.3f} pos={float(metrics['pos']):.1f}",
                    flush=True,
                )
            if step >= steps_per_epoch:
                break
        # Evaluate PR/mAP with progress bar on validation
        print("Validating...")
        # Optionally limit val_ds to args.val_steps using take()
        val_iter = val_ds.take(steps_per_val) if steps_per_val else val_ds
        p, r, m50, m5095 = evaluate_dataset_pr_maps_fast(
            model, val_iter, num_classes, conf_thres=0.001, iou_thres=0.5, max_det=300, imgsz=args.imgsz,
            progbar=None, total_steps=None,
        )
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

