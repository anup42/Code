import os
import glob
import yaml
import numpy as np
import tensorflow as tf


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images_from_dir(path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(path, "**", e), recursive=True))
    return sorted(files)


def img2label_path(img_path):
    parts = img_path.replace("\\", "/").split("/")
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
    except ValueError:
        # Fallback: replace last directory if not standard
        pass
    base, _ = os.path.splitext(parts[-1])
    parts[-1] = base + ".txt"
    return "/".join(parts)


def letterbox(image, new_shape=640, color=114, stride=32):
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    new_h, new_w = new_shape

    scale = tf.minimum(tf.cast(new_h, tf.float32) / tf.cast(h, tf.float32),
                       tf.cast(new_w, tf.float32) / tf.cast(w, tf.float32))
    nh = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    nw = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)

    resized = tf.image.resize(image, (nh, nw), method=tf.image.ResizeMethod.BILINEAR)
    pad_h = new_h - nh
    pad_w = new_w - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = tf.pad(resized, [[top, bottom], [left, right], [0, 0]], constant_values=color)
    return padded, scale, top, left


def xywhn_to_xyxy(xywhn, img_h, img_w):
    cx = xywhn[:, 0] * img_w
    cy = xywhn[:, 1] * img_h
    w = xywhn[:, 2] * img_w
    h = xywhn[:, 3] * img_h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return tf.stack([x1, y1, x2, y2], axis=-1)


def bbox_iou_xyxy(boxes1, boxes2, eps=1e-7):
    # boxes1: [N, 4], boxes2: [M, 4]
    b1 = tf.expand_dims(boxes1, 1)
    b2 = tf.expand_dims(boxes2, 0)
    inter_x1 = tf.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = tf.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = tf.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = tf.minimum(b1[..., 3], b2[..., 3])
    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter + eps
    return inter / union


def nms(boxes, scores, iou_threshold=0.5, max_detections=300):
    selected = tf.image.non_max_suppression(
        boxes, scores, max_output_size=max_detections, iou_threshold=iou_threshold
    )
    return selected

