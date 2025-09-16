import argparse
import os

import tensorflow as tf

from yolo11_tf.config import load_config
from yolo11_tf.data import build_trainer_dataset
from yolo11_tf.metrics import evaluate_dataset_map50
from yolo11_tf.model import build_yolo11
from yolo11_tf.train_utils import TrainConfig, Trainer


def parse_args():
    ap = argparse.ArgumentParser("Train YOLO11-TF (Ultralytics-style head) with DFL")
    ap.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--cfg", type=str, default=None, help="Optional path to a YAML config overriding defaults")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model_scale", type=str, default="n", choices=["n","s","m","l","x"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", type=str, default="runs/train_ultra/exp")
    ap.add_argument("--resume", type=str, default="", help="Checkpoint path or directory to resume from ('auto' to use output dir)")
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
    cfg_bundle = load_config(args.cfg)
    train_settings = cfg_bundle.train
    loss_settings = cfg_bundle.loss
    aug_cfg = cfg_bundle.augmentation

    imgsz = args.imgsz if args.imgsz is not None else train_settings.imgsz
    batch = args.batch if args.batch is not None else train_settings.batch
    epochs = args.epochs if args.epochs is not None else train_settings.epochs
    lr = args.lr if args.lr is not None else train_settings.lr0

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
    train_ds, num_classes = build_trainer_dataset(
        args.data,
        imgsz=imgsz,
        batch_size=batch,
        split="train",
        shuffle=True,
        augmentation=aug_cfg,
        augment=True,
    )
    val_ds, _ = build_trainer_dataset(
        args.data,
        imgsz=imgsz,
        batch_size=batch,
        split="val",
        shuffle=False,
        augmentation=aug_cfg,
        augment=False,
    )
    # Steps per epoch
    from yolo11_tf.data import load_yolo_yaml, build_file_list
    train_dir, val_dir, _ = load_yolo_yaml(args.data)
    train_files = build_file_list(train_dir)
    val_files = build_file_list(val_dir)
    estimated_steps = max(1, len(train_files) // batch)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch and args.steps_per_epoch > 0 else estimated_steps
    if steps_per_epoch > estimated_steps:
        train_ds = train_ds.repeat()
    steps_per_val = max(1, len(val_files) // batch)
    if args.val_steps and args.val_steps > 0:
        steps_per_val = args.val_steps

    # Model
    width_mult, depth_mult = scale_to_multipliers(args.model_scale)
    model = build_yolo11(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult, reg_max=16)

    # Trainer
    train_cfg = TrainConfig(
        num_classes=num_classes,
        img_size=imgsz,
        reg_max=16,
        lr=lr,
        weight_decay=train_settings.weight_decay,
        warmup_epochs=train_settings.warmup_epochs,
        box_loss_gain=loss_settings.box,
        cls_loss_gain=loss_settings.cls,
        dfl_loss_gain=loss_settings.dfl,
        epochs=epochs,
    )
    trainer = Trainer(model, train_cfg)

    ckpt_dir = os.path.join(args.out, "ckpt")
    ckpt_manager = tf.train.CheckpointManager(trainer.ckpt, ckpt_dir, max_to_keep=5)
    initial_epoch = 0
    resume_target = args.resume.strip()
    if resume_target:
        if resume_target.lower() == "auto":
            resume_target = ckpt_dir
        ckpt_path = None
        if os.path.isdir(resume_target):
            ckpt_path = tf.train.latest_checkpoint(resume_target)
        elif tf.io.gfile.exists(resume_target):
            ckpt_path = resume_target
        if ckpt_path:
            trainer.restore(ckpt_path)
            initial_epoch = trainer.current_epoch()
            print(
                f"Resumed from {ckpt_path} at epoch {initial_epoch} (global_step={trainer.global_step_value()})",
                flush=True,
            )
        else:
            print(f"No checkpoint found at {resume_target}, starting fresh", flush=True)

    if initial_epoch >= epochs:
        print(f"Checkpoint already completed {initial_epoch} epochs (>= {epochs}). Nothing to do.", flush=True)
        return

    # Simple training loop
    for epoch in range(initial_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")
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
        val_iter = val_ds.take(steps_per_val).map(
            lambda img, tgt: ((img, tf.zeros([tf.shape(tgt)[0], train_cfg.max_boxes, 5], dtype=tf.float32)), tgt)
        )
        p, r, m = evaluate_dataset_map50(model, val_iter, num_classes, conf_thres=0.05, iou_thres=0.5, imgsz=imgsz)
        dt = time.time() - t0
        ips = (seen / dt) if dt > 0 else 0.0
        print(
            f"Epoch {epoch+1}/{epochs} done in {dt:.1f}s ({ips:.1f} img/s) - "
            f"last_loss={metrics['loss']:.3f}  P={p:.4f} R={r:.4f} mAP50={m:.4f}",
            flush=True,
        )
        print(f"Epoch {epoch+1}/{epochs} — loss={metrics['loss']:.2f}  P={p:.4f} R={r:.4f} mAP50={m:.4f}")

        trainer.assign_epoch(epoch + 1)
        save_path = ckpt_manager.save(checkpoint_number=epoch + 1)
        if save_path:
            print(f"Saved checkpoint: {save_path}", flush=True)


if __name__ == "__main__":
    main()
