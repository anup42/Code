import argparse
import os
import glob

import numpy as np
import tensorflow as tf
from PIL import Image

from yolo11_tf.model import build_yolo11
from yolo11_tf.inference import YoloInferencer
from yolo11_tf.utils import letterbox


def parse_args():
    ap = argparse.ArgumentParser(description="Run inference with TensorFlow YOLO11-style model")
    ap.add_argument("--weights", required=True, help="Path to weights .h5")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", type=int, required=True)
    ap.add_argument("--source", type=str, required=True, help="Image file or directory")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    return ap.parse_args()


def load_images(source: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    if os.path.isdir(source):
        for ext in exts:
            paths.extend(glob.glob(os.path.join(source, f"**/*{ext}"), recursive=True))
    else:
        paths = [source]
    return sorted(paths)


def main():
    args = parse_args()
    model = build_yolo11(num_classes=args.classes, img_size=args.imgsz)
    model.load_weights(args.weights)
    infer = YoloInferencer(model, score_thresh=args.conf, iou_thresh=args.iou)

    paths = load_images(args.source)
    for p in paths:
        img0 = np.array(Image.open(p).convert("RGB"))
        img, _, _ = letterbox(img0, args.imgsz)
        x = (img.astype(np.float32) / 255.0)[None, ...]
        boxes, scores, classes, valid = infer.predict(tf.convert_to_tensor(x))
        n = int(valid[0].numpy())
        print(p)
        for i in range(n):
            b = boxes[0, i].numpy()
            s = scores[0, i].numpy()
            c = int(classes[0, i].numpy())
            print(f"  cls={c} conf={s:.3f} box={b.tolist()}")


if __name__ == "__main__":
    main()

