import argparse
import os
import time

import tensorflow as tf

from yolo11_tf.model import build_yolo11
from yolo11_tf.data import build_dataset
from yolo11_tf.losses import YoloLoss


def parse_args():
    ap = argparse.ArgumentParser("Quick training/inference throughput benchmark")
    ap.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50, help="Number of training steps to time")
    ap.add_argument("--model_scale", type=str, default="n", choices=["n","s","m","l","x"], help="Model size multiplier")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=2, help="Warmup steps before timing")
    ap.add_argument("--mode", type=str, default="train", choices=["train","infer"], help="Benchmark training step or forward-only")
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

    ds, num_classes = build_dataset(args.data, imgsz=args.imgsz, batch_size=args.batch, split="train", shuffle=True)
    width_mult, depth_mult = scale_to_multipliers(args.model_scale)
    model = build_yolo11(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult)

    loss_fn = YoloLoss(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    @tf.function(jit_compile=False)
    def train_step(images, targets):
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss = loss_fn(targets, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function(jit_compile=False)
    def infer_step(images):
        _ = model(images, training=False)

    # Warmup
    it = iter(ds)
    for i in range(args.warmup):
        images, targets = next(it)
        if args.mode == "train":
            _ = train_step(images, targets)
        else:
            infer_step(images)

    # Timed loop
    steps = 0
    start = time.perf_counter()
    for images, targets in ds:
        if args.mode == "train":
            _ = train_step(images, targets)
        else:
            infer_step(images)
        steps += 1
        if steps >= args.steps:
            break
    total = time.perf_counter() - start

    imgs = steps * args.batch
    print(f"mode={args.mode} steps={steps} batch={args.batch} imgsz={args.imgsz}")
    print(f"total_time_sec={total:.3f}")
    if steps > 0 and total > 0:
        print(f"avg_step_ms={(total/steps)*1000:.2f}")
        print(f"imgs_per_sec={imgs/total:.2f}")


if __name__ == "__main__":
    main()

