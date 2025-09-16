import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from .config import AugmentationConfig
from .utils import img2label_path, letterbox, list_images_from_dir, read_yaml


@dataclass
class DatasetConfig:
    imgsz: int = 640
    batch_size: int = 16
    num_classes: int = 80
    # Strides should match model output downsampling ratios
    strides: Tuple[int, int, int] = (4, 8, 16)
    max_labels: int = 300  # per image
    augmentation: Optional[AugmentationConfig] = None


def load_yolo_yaml(data_yaml_path: str):
    data = read_yaml(data_yaml_path)
    # typical keys: path, train, val, test, names
    base = data.get("path", None)
    def _resolve(p):
        if p is None:
            return None
        if base is not None and not os.path.isabs(p):
            return os.path.join(base, p)
        return p

    train = _resolve(data.get("train"))
    val = _resolve(data.get("val"))
    names = data.get("names")
    if isinstance(names, dict):
        num_classes = len(names)
    else:
        num_classes = len(names) if names is not None else data.get("nc", 80)
    return train, val, num_classes


def build_file_list(img_dir_or_txt: str) -> List[str]:
    if os.path.isdir(img_dir_or_txt):
        return list_images_from_dir(img_dir_or_txt)
    if os.path.isfile(img_dir_or_txt):
        # Could be a txt containing image paths
        with open(img_dir_or_txt, "r", encoding="utf-8") as f:
            return [x.strip() for x in f.readlines() if x.strip()]
    raise FileNotFoundError(f"Couldn't resolve dataset path: {img_dir_or_txt}")


def read_label_file(label_path: str) -> np.ndarray:
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=np.float32)
    lines = []
    with open(label_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                continue
            lines.append([float(p) for p in parts])
    if not lines:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(lines, dtype=np.float32)


def _preprocess_basic(img_path, cfg: DatasetConfig):
    # img_path is a tf.string Tensor; use TF ops for image IO
    # Load image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    h0, w0 = tf.shape(img)[0], tf.shape(img)[1]

    # Load labels via numpy (py_function). Compute label path in Python inside the function.
    def _read_labels(p):
        # p is a numpy bytes_ scalar from TF; decode and map to labels path
        if isinstance(p, (bytes, bytearray)):
            p_str = p.decode("utf-8")
        else:
            p_str = str(p)
        lpath = img2label_path(p_str)
        return read_label_file(lpath).astype(np.float32)

    labels_np = tf.numpy_function(_read_labels, [img_path], tf.float32)
    labels_np.set_shape([None, 5])

    # Letterbox
    lb_img, scale, pad_top, pad_left = letterbox(img, new_shape=cfg.imgsz)
    new_h, new_w = cfg.imgsz, cfg.imgsz

    # Adjust labels: [cls, cx, cy, w, h] normalized to new image size
    if tf.shape(labels_np)[0] > 0:
        cls = labels_np[:, 0]
        xywhn = labels_np[:, 1:5]
        # Convert normalized to absolute original, then scale and pad
        cx = xywhn[:, 0] * tf.cast(w0, tf.float32)
        cy = xywhn[:, 1] * tf.cast(h0, tf.float32)
        ww = xywhn[:, 2] * tf.cast(w0, tf.float32)
        hh = xywhn[:, 3] * tf.cast(h0, tf.float32)
        cx = cx * scale + tf.cast(pad_left, tf.float32)
        cy = cy * scale + tf.cast(pad_top, tf.float32)
        ww = ww * scale
        hh = hh * scale
        cxn = cx / float(new_w)
        cyn = cy / float(new_h)
        wwn = ww / float(new_w)
        hhn = hh / float(new_h)
        labels_adj = tf.stack([cls, cxn, cyn, wwn, hhn], axis=-1)
    else:
        labels_adj = tf.zeros((0, 5), dtype=tf.float32)

    # Pad/clip labels to fixed length per image for batching
    max_labels = cfg.max_labels
    n = tf.shape(labels_adj)[0]
    k = tf.minimum(n, max_labels)
    labels_adj = labels_adj[:k]
    pad_rows = tf.maximum(0, max_labels - k)
    labels_adj = tf.pad(labels_adj, [[0, pad_rows], [0, 0]])

    # Normalize image 0-1
    lb_img = lb_img / 255.0
    return lb_img, labels_adj


# -----------------------------------------------------------------------------
# Augmentation helpers inspired by Ultralytics YOLO11
# -----------------------------------------------------------------------------


def _read_image(path: str) -> np.ndarray:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    return img.numpy()


def _read_image_and_labels(path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = _read_image(path)
    labels = read_label_file(img2label_path(path)).astype(np.float32)
    if labels.size == 0:
        return img, np.zeros((0, 5), dtype=np.float32)
    h, w = img.shape[:2]
    cls = labels[:, 0:1]
    xywh = labels[:, 1:5]
    x = xywh[:, 0] * w
    y = xywh[:, 1] * h
    bw = xywh[:, 2] * w
    bh = xywh[:, 3] * h
    x1 = x - bw / 2.0
    y1 = y - bh / 2.0
    x2 = x + bw / 2.0
    y2 = y + bh / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    labels_xyxy = np.concatenate([cls, boxes], axis=1).astype(np.float32)
    return img, labels_xyxy


def _letterbox_image_np(image: np.ndarray, imgsz: int, pad_val: float) -> Tuple[np.ndarray, float, int, int]:
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    lb_img, scale, pad_top, pad_left = letterbox(tensor, new_shape=imgsz, color=pad_val)
    return lb_img.numpy(), float(scale.numpy()), int(pad_top.numpy()), int(pad_left.numpy())


def _load_image_and_labels_xyxy(path: str, cfg: DatasetConfig, aug_cfg: AugmentationConfig) -> Tuple[np.ndarray, np.ndarray]:
    img, labels = _read_image_and_labels(path)
    img_lb, scale, pad_top, pad_left = _letterbox_image_np(img, cfg.imgsz, aug_cfg.pad_val)
    if labels.size:
        labels = labels.copy()
        labels[:, [1, 3]] = labels[:, [1, 3]] * scale + pad_left
        labels[:, [2, 4]] = labels[:, [2, 4]] * scale + pad_top
    return img_lb, labels


def _load_mosaic_image(index: int, files: List[str], cfg: DatasetConfig, aug_cfg: AugmentationConfig) -> Tuple[np.ndarray, np.ndarray]:
    s = cfg.imgsz
    mosaic_img = np.full((s * 2, s * 2, 3), aug_cfg.pad_val, dtype=np.float32)
    mosaic_labels = []
    yc = random.randint(int(0.5 * s), int(1.5 * s))
    xc = random.randint(int(0.5 * s), int(1.5 * s))
    indices = [index] + random.choices(range(len(files)), k=3)

    for i, idx in enumerate(indices):
        path = files[int(idx)]
        img, labels = _read_image_and_labels(path)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            continue
        scale = random.uniform(*aug_cfg.mosaic_scale)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        img_resized = tf.image.resize(img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR).numpy()

        if i == 0:  # top left
            x1a = max(xc - new_w, 0)
            y1a = max(yc - new_h, 0)
            x2a = xc
            y2a = yc
            x1b = new_w - (x2a - x1a)
            y1b = new_h - (y2a - y1a)
            x2b = new_w
            y2b = new_h
        elif i == 1:  # top right
            x1a = xc
            y1a = max(yc - new_h, 0)
            x2a = min(xc + new_w, s * 2)
            y2a = yc
            x1b = 0
            y1b = new_h - (y2a - y1a)
            x2b = min(new_w, x2a - x1a)
            y2b = new_h
        elif i == 2:  # bottom left
            x1a = max(xc - new_w, 0)
            y1a = yc
            x2a = xc
            y2a = min(s * 2, yc + new_h)
            x1b = new_w - (x2a - x1a)
            y1b = 0
            x2b = new_w
            y2b = min(new_h, y2a - y1a)
        else:  # bottom right
            x1a = xc
            y1a = yc
            x2a = min(xc + new_w, s * 2)
            y2a = min(yc + new_h, s * 2)
            x1b = 0
            y1b = 0
            x2b = min(new_w, x2a - x1a)
            y2b = min(new_h, y2a - y1a)

        mosaic_img[y1a:y2a, x1a:x2a] = img_resized[y1b:y2b, x1b:x2b]

        if labels.size:
            labels = labels.copy()
            labels[:, [1, 3]] *= new_w / w
            labels[:, [2, 4]] *= new_h / h
            labels[:, [1, 3]] += x1a - x1b
            labels[:, [2, 4]] += y1a - y1b
            mosaic_labels.append(labels)

    if mosaic_labels:
        labels = np.concatenate(mosaic_labels, axis=0)
    else:
        labels = np.zeros((0, 5), dtype=np.float32)
    return mosaic_img, labels


def _warp_image_projective(image: np.ndarray, matrix: np.ndarray, size: Tuple[int, int], pad_val: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    output_shape = tf.constant([int(size[1]), int(size[0])], dtype=tf.int32)
    image_tf = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    matrix_inv = np.linalg.inv(matrix)
    transform = [
        matrix_inv[0, 0],
        matrix_inv[0, 1],
        matrix_inv[0, 2],
        matrix_inv[1, 0],
        matrix_inv[1, 1],
        matrix_inv[1, 2],
        matrix_inv[2, 0],
        matrix_inv[2, 1],
    ]
    transform_tf = tf.convert_to_tensor([transform], dtype=tf.float32)
    warped = tf.raw_ops.ImageProjectiveTransformV3(
        images=image_tf,
        transforms=transform_tf,
        output_shape=output_shape,
        interpolation="BILINEAR",
        fill_value=float(pad_val),
    )
    return warped[0].numpy()


def _apply_boxes(boxes: np.ndarray, matrix: np.ndarray, perspective: bool) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    n = boxes.shape[0]
    corners = np.ones((n * 4, 3), dtype=np.float32)
    corners[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(-1, 2)
    transformed = corners @ matrix.T
    if perspective:
        transformed = transformed[:, :2] / transformed[:, 2:3]
    else:
        transformed = transformed[:, :2]
    transformed = transformed.reshape(n, 8)
    x = transformed[:, [0, 2, 4, 6]]
    y = transformed[:, [1, 3, 5, 7]]
    return np.stack([x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)], axis=1)


def _clip_boxes(boxes: np.ndarray, width: float, height: float) -> None:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height)


def _box_candidates(boxes_old: np.ndarray, boxes_new: np.ndarray, wh_thr: float = 2.0, ar_thr: float = 100.0, area_thr: float = 0.1) -> np.ndarray:
    if boxes_new.size == 0:
        return np.zeros((0,), dtype=bool)
    w1 = boxes_old[:, 2] - boxes_old[:, 0]
    h1 = boxes_old[:, 3] - boxes_old[:, 1]
    w2 = boxes_new[:, 2] - boxes_new[:, 0]
    h2 = boxes_new[:, 3] - boxes_new[:, 1]
    area1 = w1 * h1 + 1e-9
    area2 = w2 * h2
    aspect = np.maximum(w2 / (h2 + 1e-9), h2 / (w2 + 1e-9))
    mask = (w2 > wh_thr) & (h2 > wh_thr) & (area2 / area1 > area_thr) & (aspect < ar_thr)
    mask &= (w2 > 0) & (h2 > 0)
    return mask


def _filter_valid_boxes(labels: np.ndarray, width: float, height: float) -> np.ndarray:
    if labels.size == 0:
        return labels.astype(np.float32)
    boxes = labels[:, 1:5].copy()
    _clip_boxes(boxes, width, height)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = (w > 1.0) & (h > 1.0)
    labels = labels[mask]
    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    labels[:, 1:5] = boxes[mask]
    return labels.astype(np.float32)


def _labels_xyxy_to_normalized(labels: np.ndarray, width: float, height: float) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    boxes = labels[:, 1:5]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / max(width, 1e-6)
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / max(height, 1e-6)
    bw = (boxes[:, 2] - boxes[:, 0]) / max(width, 1e-6)
    bh = (boxes[:, 3] - boxes[:, 1]) / max(height, 1e-6)
    cx = np.clip(cx, 0.0, 1.0)
    cy = np.clip(cy, 0.0, 1.0)
    bw = np.clip(bw, 0.0, 1.0)
    bh = np.clip(bh, 0.0, 1.0)
    return np.concatenate([labels[:, :1], np.stack([cx, cy, bw, bh], axis=1)], axis=1).astype(np.float32)


def _augment_hsv_np(image: np.ndarray, aug_cfg: AugmentationConfig) -> np.ndarray:
    if aug_cfg.hsv_h == 0 and aug_cfg.hsv_s == 0 and aug_cfg.hsv_v == 0:
        return image
    gains = np.random.uniform(-1, 1, size=3) * np.array([aug_cfg.hsv_h, aug_cfg.hsv_s, aug_cfg.hsv_v], dtype=np.float32)
    img_tf = tf.convert_to_tensor(image / 255.0, dtype=tf.float32)
    hsv = tf.image.rgb_to_hsv(img_tf)
    h, s, v = tf.unstack(hsv, axis=-1)
    h = tf.math.floormod(h + gains[0], 1.0)
    s = tf.clip_by_value(s * (1.0 + gains[1]), 0.0, 1.0)
    v = tf.clip_by_value(v * (1.0 + gains[2]), 0.0, 1.0)
    hsv_aug = tf.stack([h, s, v], axis=-1)
    rgb = tf.image.hsv_to_rgb(hsv_aug)
    return (rgb * 255.0).numpy()


def _random_flip_np(image: np.ndarray, labels: np.ndarray, aug_cfg: AugmentationConfig) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    boxes = labels[:, 1:5].copy() if labels.size else labels[:, 1:5]
    if aug_cfg.flipud > 0 and labels.size and random.random() < aug_cfg.flipud:
        image = np.flipud(image)
        y1 = boxes[:, 1].copy()
        y2 = boxes[:, 3].copy()
        boxes[:, 1] = h - y2
        boxes[:, 3] = h - y1
    if aug_cfg.fliplr > 0 and labels.size and random.random() < aug_cfg.fliplr:
        image = np.fliplr(image)
        x1 = boxes[:, 0].copy()
        x2 = boxes[:, 2].copy()
        boxes[:, 0] = w - x2
        boxes[:, 2] = w - x1
    if labels.size:
        labels = labels.copy()
        labels[:, 1:5] = boxes
    return image, labels


def _random_affine(image: np.ndarray, labels: np.ndarray, cfg: DatasetConfig, aug_cfg: AugmentationConfig, border: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    size = (int(round(w + border[1] * 2)), int(round(h + border[0] * 2)))
    C = np.eye(3, dtype=np.float32)
    C[0, 2] = -w / 2.0
    C[1, 2] = -h / 2.0

    P = np.eye(3, dtype=np.float32)
    P[2, 0] = random.uniform(-aug_cfg.perspective, aug_cfg.perspective)
    P[2, 1] = random.uniform(-aug_cfg.perspective, aug_cfg.perspective)

    R = np.eye(3, dtype=np.float32)
    angle = random.uniform(-aug_cfg.degrees, aug_cfg.degrees)
    scale = random.uniform(1 - aug_cfg.scale, 1 + aug_cfg.scale)
    rad = math.radians(angle)
    sa, ca = math.sin(rad), math.cos(rad)
    R[0, 0] = ca * scale
    R[0, 1] = sa * scale
    R[1, 0] = -sa * scale
    R[1, 1] = ca * scale

    S = np.eye(3, dtype=np.float32)
    S[0, 1] = math.tan(math.radians(random.uniform(-aug_cfg.shear, aug_cfg.shear)))
    S[1, 0] = math.tan(math.radians(random.uniform(-aug_cfg.shear, aug_cfg.shear)))

    T = np.eye(3, dtype=np.float32)
    T[0, 2] = random.uniform(0.5 - aug_cfg.translate, 0.5 + aug_cfg.translate) * size[0]
    T[1, 2] = random.uniform(0.5 - aug_cfg.translate, 0.5 + aug_cfg.translate) * size[1]

    M = T @ S @ R @ P @ C
    warped = _warp_image_projective(image, M, size, aug_cfg.pad_val)

    if labels.size:
        boxes = labels[:, 1:5].copy()
        new_boxes = _apply_boxes(boxes, M, aug_cfg.perspective > 0)
        _clip_boxes(new_boxes, size[0], size[1])
        mask = _box_candidates(boxes, new_boxes)
        labels = labels[mask]
        new_boxes = new_boxes[mask]
        labels = labels.astype(np.float32)
        if labels.size:
            labels[:, 1:5] = new_boxes
        else:
            labels = np.zeros((0, 5), dtype=np.float32)
    else:
        labels = np.zeros((0, 5), dtype=np.float32)
    return warped, labels


def _load_augmented_xyxy(index: int, path: str, cfg: DatasetConfig, files: List[str], aug_cfg: AugmentationConfig, allow_mixup: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    use_mosaic = aug_cfg.mosaic > 0 and len(files) >= 4 and random.random() < aug_cfg.mosaic
    if use_mosaic:
        img, labels = _load_mosaic_image(index, files, cfg, aug_cfg)
        img, labels = _random_affine(img, labels, cfg, aug_cfg, border=(-cfg.imgsz // 2, -cfg.imgsz // 2))
    else:
        img, labels = _load_image_and_labels_xyxy(path, cfg, aug_cfg)
        img, labels = _random_affine(img, labels, cfg, aug_cfg, border=(0, 0))

    if allow_mixup and aug_cfg.mixup > 0 and len(files) > 1 and random.random() < aug_cfg.mixup:
        mix_index = random.randint(0, len(files) - 1)
        mix_path = files[mix_index]
        mix_img, mix_labels = _load_augmented_xyxy(mix_index, mix_path, cfg, files, aug_cfg, allow_mixup=False)
        ratio = np.random.beta(aug_cfg.mixup_beta, aug_cfg.mixup_beta)
        img = img * ratio + mix_img * (1.0 - ratio)
        if labels.size and mix_labels.size:
            labels = np.concatenate([labels, mix_labels], axis=0)
        elif mix_labels.size:
            labels = mix_labels
    return img, labels


def _pad_labels(labels: np.ndarray, max_labels: int) -> np.ndarray:
    labels = labels.astype(np.float32)
    n = labels.shape[0]
    if n >= max_labels:
        return labels[:max_labels]
    if n == 0:
        return np.zeros((max_labels, 5), dtype=np.float32)
    pad = np.zeros((max_labels - n, 5), dtype=np.float32)
    return np.concatenate([labels, pad], axis=0)


def _load_augmented_example(index: int, path: str, cfg: DatasetConfig, files: List[str], aug_cfg: AugmentationConfig) -> Tuple[np.ndarray, np.ndarray]:
    img, labels_xyxy = _load_augmented_xyxy(index, path, cfg, files, aug_cfg)
    img = _augment_hsv_np(img, aug_cfg)
    img, labels_xyxy = _random_flip_np(img, labels_xyxy, aug_cfg)
    labels_xyxy = _filter_valid_boxes(labels_xyxy, img.shape[1], img.shape[0])
    labels_norm = _labels_xyxy_to_normalized(labels_xyxy, img.shape[1], img.shape[0])
    return img.astype(np.float32), labels_norm


def _preprocess_augmented(idx, img_path, cfg: DatasetConfig, files: List[str]):
    aug_cfg = cfg.augmentation
    if aug_cfg is None:
        raise ValueError("Augmentation config must be provided for augmented preprocessing")

    def _py(index_np, path_np):
        index = int(index_np)
        if isinstance(path_np, (bytes, bytearray)):
            path = path_np.decode("utf-8")
        else:
            path = str(path_np)
        img, labels = _load_augmented_example(index, path, cfg, files, aug_cfg)
        img = np.clip(img, 0.0, 255.0)
        labels = _pad_labels(labels, cfg.max_labels)
        return img / 255.0, labels.astype(np.float32)

    image, labels = tf.numpy_function(_py, [idx, img_path], [tf.float32, tf.float32])
    image.set_shape((cfg.imgsz, cfg.imgsz, 3))
    labels.set_shape((cfg.max_labels, 5))
    return image, labels


def targets_for_strides(labels, cfg: DatasetConfig):
    # Prepare targets per scale: [H, W, 5 + C]
    num_classes = cfg.num_classes
    out = []
    for s in cfg.strides:
        gs = cfg.imgsz // s
        out.append(np.zeros((gs, gs, 5 + num_classes), dtype=np.float32))

    # Assign each label to its center cell at the appropriate scale based on box size
    for lab in labels:
        c = int(lab[0])
        cx, cy, w, h = lab[1:5]
        abs_w = w * cfg.imgsz
        abs_h = h * cfg.imgsz
        scale = max(abs_w, abs_h)
        # pick scale heuristically
        if scale <= 64:
            scales = [0]
        elif scale <= 128:
            scales = [0, 1]
        elif scale <= 256:
            scales = [1]
        else:
            scales = [2]
        for si in scales:
            s = cfg.strides[si]
            gs = cfg.imgsz // s
            gi = int(cx * gs)
            gj = int(cy * gs)
            gi = np.clip(gi, 0, gs - 1)
            gj = np.clip(gj, 0, gs - 1)
            target = out[si]
            # objectness
            target[gj, gi, 4] = 1.0
            # class one-hot
            target[gj, gi, 5 + c] = 1.0
            # box: cx, cy, w, h normalized to feature map grid (relative to image)
            target[gj, gi, 0:4] = np.array([cx, cy, w, h], dtype=np.float32)
    return out


def make_example_with_targets(idx: tf.Tensor, img_path: tf.Tensor, cfg: DatasetConfig, files: List[str], augment: bool):
    if augment and cfg.augmentation is not None:
        img, labels = _preprocess_augmented(idx, img_path, cfg, files)
    else:
        img, labels = _preprocess_basic(img_path, cfg)
    # Convert to numpy and create targets with py_function to keep tf.data graph simple
    def _to_targets(lbls):
        lbls = lbls.astype(np.float32)
        return [targets_for_strides(lbls, cfg)[i] for i in range(3)]

    t0, t1, t2 = tf.numpy_function(_to_targets, [labels], [tf.float32, tf.float32, tf.float32])
    t0.set_shape((cfg.imgsz // cfg.strides[0], cfg.imgsz // cfg.strides[0], 5 + cfg.num_classes))
    t1.set_shape((cfg.imgsz // cfg.strides[1], cfg.imgsz // cfg.strides[1], 5 + cfg.num_classes))
    t2.set_shape((cfg.imgsz // cfg.strides[2], cfg.imgsz // cfg.strides[2], 5 + cfg.num_classes))
    # Also return normalized labels [N,5] with columns [cls, cx, cy, w, h]
    return img, (t0, t1, t2), labels


def build_dataset(
    data_yaml: str,
    imgsz=640,
    batch_size=16,
    split="train",
    shuffle=True,
    num_parallel_calls=tf.data.AUTOTUNE,
    include_labels=True,
    augmentation: Optional[AugmentationConfig] = None,
    augment: Optional[bool] = None,
):
    train, val, num_classes = load_yolo_yaml(data_yaml)
    src = train if split == "train" else val
    files = build_file_list(src)
    if len(files) == 0:
        raise ValueError(f"No images found for split '{split}' at {src}")
    file_list = list(files)
    indices = np.arange(len(file_list), dtype=np.int32)
    files_np = np.array(file_list, dtype=np.string_)

    use_aug = augment if augment is not None else (split == "train" and augmentation is not None)
    cfg = DatasetConfig(
        imgsz=imgsz,
        batch_size=batch_size,
        num_classes=num_classes,
        augmentation=augmentation if use_aug else None,
    )

    ds = tf.data.Dataset.from_tensor_slices((indices, files_np))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(file_list)))
    ds = ds.map(
        lambda i, p: make_example_with_targets(i, p, cfg, file_list, use_aug),
        num_parallel_calls=num_parallel_calls,
    )
    # Repack to include labels for metrics while keeping targets as y
    def _pack(img, tlist, labels):
        if include_labels:
            return (img, labels), tlist
        else:
            return img, tlist
    ds = ds.map(_pack, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, num_classes


def build_trainer_dataset(
    data_yaml: str,
    imgsz=640,
    batch_size=16,
    split="train",
    shuffle=True,
    num_parallel_calls=tf.data.AUTOTUNE,
    augmentation: Optional[AugmentationConfig] = None,
    augment: Optional[bool] = None,
):
    """Dataset for Trainer pipeline: returns (images, targets) where
    targets is [max_labels, 6] with [cls, x1, y1, x2, y2, valid] in pixel coords.
    """
    train, val, num_classes = load_yolo_yaml(data_yaml)
    src = train if split == "train" else val
    files = build_file_list(src)
    if len(files) == 0:
        raise ValueError(f"No images found for split '{split}' at {src}")
    file_list = list(files)
    indices = np.arange(len(file_list), dtype=np.int32)
    files_np = np.array(file_list, dtype=np.string_)

    use_aug = augment if augment is not None else (split == "train" and augmentation is not None)
    cfg = DatasetConfig(
        imgsz=imgsz,
        batch_size=batch_size,
        num_classes=num_classes,
        augmentation=augmentation if use_aug else None,
    )

    def _make(idx: tf.Tensor, img_path: tf.Tensor):
        if use_aug and cfg.augmentation is not None:
            img, labels = _preprocess_augmented(idx, img_path, cfg, file_list)
        else:
            img, labels = _preprocess_basic(img_path, cfg)
        # labels: [max_labels,5] [cls,cx,cy,w,h] normalized
        cls = labels[:, 0:1]
        cx = labels[:, 1:2]
        cy = labels[:, 2:3]
        w = labels[:, 3:4]
        h = labels[:, 4:5]
        x1 = (cx - 0.5 * w) * float(cfg.imgsz)
        y1 = (cy - 0.5 * h) * float(cfg.imgsz)
        x2 = (cx + 0.5 * w) * float(cfg.imgsz)
        y2 = (cy + 0.5 * h) * float(cfg.imgsz)
        valid = tf.cast((w > 0) & (h > 0), tf.float32)
        t = tf.concat([cls, x1, y1, x2, y2, valid], axis=-1)  # [max_labels,6]
        return img, t

    ds = tf.data.Dataset.from_tensor_slices((indices, files_np))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(file_list)))
    ds = ds.map(lambda i, p: _make(i, p), num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, num_classes


# =====================
# TFRecord support
# =====================

def _serialize_example(img_path: str):
    """Read an image path + its YOLO label file and return serialized TF Example.

    Stores:
      - image: bytes (encoded original image)
      - shape: int64[2] (h, w)
      - labels: float32 list (flattened [N,5] rows [cls,cx,cy,w,h])
    """
    from io import BytesIO
    from PIL import Image  # light dependency, commonly available; avoids cv2 requirement
    p = img_path
    with open(p, 'rb') as f:
        img_bytes = f.read()
    # Determine shape by decoding headers
    with Image.open(BytesIO(img_bytes)) as im:
        w, h = im.size
    labels = read_label_file(img2label_path(p)).astype(np.float32)
    labels_flat = labels.reshape(-1).astype(np.float32)

    def _bytes_feature(v: bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    def _int64_list(v):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

    def _float_list(v):
        return tf.train.Feature(float_list=tf.train.FloatList(value=v))

    ex = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(img_bytes),
        'shape': _int64_list([int(h), int(w)]),
        'labels': _float_list(labels_flat.tolist()),
    }))
    return ex.SerializeToString()


def _serialize_example_idx(args):
    i, p = args
    return i, _serialize_example(p)


def write_tfrecords_from_yaml(
    data_yaml: str,
    out_dir: str,
    split: str = 'train',
    shards: int = 8,
    update_every: int = 10,
    num_workers: int | None = None,
    use_processes: bool = False,
):
    train, val, _ = load_yolo_yaml(data_yaml)
    src = train if split == 'train' else val
    files = build_file_list(src)
    out_dir = os.path.join(out_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    n = len(files)
    per = max(1, n // shards)
    writers = []
    for s in range(shards):
        writers.append(tf.io.TFRecordWriter(os.path.join(out_dir, f"{split}-{s:03d}.tfrecord")))
    # Progress bar
    pb = None
    if n > 0:
        try:
            pb = tf.keras.utils.Progbar(n, unit_name='img')
        except Exception:
            pb = None
    import time
    t0 = time.time()
    try:
        # Parallel serialization (threads by default for portability; opt-in processes)
        if num_workers is None or num_workers <= 0:
            try:
                num_workers = max(1, (os.cpu_count() or 4) // 2)
            except Exception:
                num_workers = 4
        if num_workers == 1:
            # Sequential fallback
            for i, p in enumerate(files):
                ex = _serialize_example(p)
                writers[(i // per) % shards].write(ex)
                if pb is not None:
                    done = i + 1
                    if (done % max(1, int(update_every)) == 0) or (done == n):
                        dt = max(1e-9, time.time() - t0)
                        ips = done / dt
                        remaining = n - done
                        eta = remaining / ips if ips > 0 else 0.0
                        pb.update(done, values=[("ips", ips), ("eta_s", eta)])
        else:
            Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
            with Executor(max_workers=num_workers) as ex_pool:
                futs = [ex_pool.submit(_serialize_example_idx, (i, p)) for i, p in enumerate(files)]
                done_count = 0
                for fut in as_completed(futs):
                    i, ser = fut.result()
                    writers[(i // per) % shards].write(ser)
                    done_count += 1
                    if pb is not None and ((done_count % max(1, int(update_every)) == 0) or (done_count == n)):
                        dt = max(1e-9, time.time() - t0)
                        ips = done_count / dt
                        remaining = n - done_count
                        eta = remaining / ips if ips > 0 else 0.0
                        pb.update(done_count, values=[("ips", ips), ("eta_s", eta)])
    finally:
        for w in writers:
            w.close()
    return out_dir


def build_trainer_dataset_from_tfrecord(tfrec_glob: str, imgsz=640, batch_size=16, shuffle=True,
                                        num_parallel_calls=tf.data.AUTOTUNE, max_labels=300):
    feature_spec = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.FixedLenFeature([2], tf.int64),
        'labels': tf.io.VarLenFeature(tf.float32),
    }

    def _parse(rec):
        ex = tf.io.parse_single_example(rec, feature_spec)
        img = tf.image.decode_image(ex['image'], channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)
        h0 = tf.cast(ex['shape'][0], tf.int32)
        w0 = tf.cast(ex['shape'][1], tf.int32)
        labels = tf.sparse.to_dense(ex['labels'])
        labels = tf.reshape(labels, [-1, 5])  # [N,5]
        # Letterbox
        lb_img, scale, pad_top, pad_left = letterbox(img, new_shape=imgsz)
        new_h, new_w = imgsz, imgsz
        # Adjust labels to normalized in new image size
        if tf.shape(labels)[0] > 0:
            cls = labels[:, 0:1]
            xywhn = labels[:, 1:5]
            cx = xywhn[:, 0] * tf.cast(w0, tf.float32)
            cy = xywhn[:, 1] * tf.cast(h0, tf.float32)
            ww = xywhn[:, 2] * tf.cast(w0, tf.float32)
            hh = xywhn[:, 3] * tf.cast(h0, tf.float32)
            cx = cx * scale + tf.cast(pad_left, tf.float32)
            cy = cy * scale + tf.cast(pad_top, tf.float32)
            ww = ww * scale
            hh = hh * scale
            cxn = cx / float(new_w)
            cyn = cy / float(new_h)
            wwn = ww / float(new_w)
            hhn = hh / float(new_h)
            labels_adj = tf.stack([tf.squeeze(cls, 1), cxn, cyn, wwn, hhn], axis=-1)
        else:
            labels_adj = tf.zeros((0, 5), dtype=tf.float32)
        # Pad to fixed max_labels
        n = tf.shape(labels_adj)[0]
        k = tf.minimum(n, tf.constant(max_labels, tf.int32))
        labels_adj = labels_adj[:k]
        pad_rows = tf.maximum(0, max_labels - k)
        labels_adj = tf.pad(labels_adj, [[0, pad_rows], [0, 0]])
        # Convert to Trainer targets [max_labels,6] in pixel coords
        cls = labels_adj[:, 0:1]
        cx = labels_adj[:, 1:2]
        cy = labels_adj[:, 2:3]
        w = labels_adj[:, 3:4]
        h = labels_adj[:, 4:5]
        x1 = (cx - 0.5 * w) * float(imgsz)
        y1 = (cy - 0.5 * h) * float(imgsz)
        x2 = (cx + 0.5 * w) * float(imgsz)
        y2 = (cy + 0.5 * h) * float(imgsz)
        valid = tf.cast((w > 0) & (h > 0), tf.float32)
        t = tf.concat([cls, x1, y1, x2, y2, valid], axis=-1)
        lb_img = lb_img / 255.0
        return lb_img, t

    files = tf.io.gfile.glob(os.path.join(tfrec_glob, '*.tfrecord')) if os.path.isdir(tfrec_glob) else tf.io.gfile.glob(tfrec_glob)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=8192)
    ds = ds.map(_parse, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # Return dataset and infer num_classes from labels (user still provides via YAML typically)
    return ds
