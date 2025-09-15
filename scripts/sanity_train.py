import os
import numpy as np
import tensorflow as tf

from yolo11_tf.model import build_yolo11
from yolo11_tf.train_utils import Trainer, TrainConfig


def make_fake_batch(batch=2, img_size=640, num_classes=3, max_boxes=10):
    images = np.random.rand(batch, img_size, img_size, 3).astype(np.float32)
    targets = np.zeros((batch, max_boxes, 6), dtype=np.float32)
    for b in range(batch):
        n = np.random.randint(1, max_boxes + 1)
        for i in range(n):
            cls = np.random.randint(0, num_classes)
            x1 = np.random.uniform(0, img_size * 0.7)
            y1 = np.random.uniform(0, img_size * 0.7)
            w = np.random.uniform(5, img_size * 0.2)
            h = np.random.uniform(5, img_size * 0.2)
            x2 = min(img_size - 1, x1 + w)
            y2 = min(img_size - 1, y1 + h)
            targets[b, i, :] = [cls, x1, y1, x2, y2, 1]
    return images, targets


def main():
    img_size = 256
    num_classes = 3
    batch = 2

    cfg = TrainConfig(num_classes=num_classes, img_size=img_size, reg_max=16, lr=1e-3, max_boxes=10)
    model = build_yolo11(num_classes=num_classes, width_mult=0.25, depth_mult=0.25, reg_max=cfg.reg_max)
    trainer = Trainer(model, cfg)

    images, targets = make_fake_batch(batch=batch, img_size=img_size, num_classes=num_classes, max_boxes=cfg.max_boxes)
    images = tf.convert_to_tensor(images)
    targets = tf.convert_to_tensor(targets)

    metrics = trainer.train_step(images, targets)
    print({k: float(v.numpy()) for k, v in metrics.items()})


if __name__ == "__main__":
    main()

