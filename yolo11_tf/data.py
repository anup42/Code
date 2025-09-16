import os
import random
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import List, Tuple

from .utils import list_images_from_dir, img2label_path, read_yaml, letterbox


@dataclass
class DatasetConfig:
    imgsz: int = 640
    batch_size: int = 16
    num_classes: int = 80
    # Strides should match model output downsampling ratios
    strides: Tuple[int, int, int] = (4, 8, 16)
    max_labels: int = 300  # per image


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


def preprocess_example(img_path, cfg: DatasetConfig):
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


def make_example_with_targets(img_path: bytes, cfg: DatasetConfig):
    img, labels = preprocess_example(img_path, cfg)
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


def build_dataset(data_yaml: str, imgsz=640, batch_size=16, split="train", shuffle=True, num_parallel_calls=tf.data.AUTOTUNE, include_labels=True):
    train, val, num_classes = load_yolo_yaml(data_yaml)
    src = train if split == "train" else val
    files = build_file_list(src)
    cfg = DatasetConfig(imgsz=imgsz, batch_size=batch_size, num_classes=num_classes)

    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(files)))
    ds = ds.map(lambda p: make_example_with_targets(p, cfg), num_parallel_calls=num_parallel_calls)
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


def build_trainer_dataset(data_yaml: str, imgsz=640, batch_size=16, split="train", shuffle=True, num_parallel_calls=tf.data.AUTOTUNE):
    """Dataset for Trainer pipeline: returns (images, targets) where
    targets is [max_labels, 6] with [cls, x1, y1, x2, y2, valid] in pixel coords.
    """
    train, val, num_classes = load_yolo_yaml(data_yaml)
    src = train if split == "train" else val
    files = build_file_list(src)
    cfg = DatasetConfig(imgsz=imgsz, batch_size=batch_size, num_classes=num_classes)

    def _make(img_path: bytes):
        img, labels = preprocess_example(img_path, cfg)
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

    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(files)))
    ds = ds.map(lambda p: _make(p), num_parallel_calls=num_parallel_calls)
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
    import cv2  # local import to avoid hard dep if user doesn't use TFRecords
    p = img_path
    with open(p, 'rb') as f:
        img_bytes = f.read()
    # get shape via cv2 decode for reliability
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
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


def write_tfrecords_from_yaml(data_yaml: str, out_dir: str, split: str = 'train', shards: int = 8):
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
    try:
        for i, p in enumerate(files):
            ex = _serialize_example(p)
            writers[(i // per) % shards].write(ex)
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
