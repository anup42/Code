from typing import List, Tuple

import tensorflow as tf

from .losses import integral_distribution
from .utils import combined_nms


class YoloInferencer:
    def __init__(self, model: tf.keras.Model, score_thresh: float = 0.25, iou_thresh: float = 0.7):
        self.model = model
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        head = getattr(model, "head", None)
        self.reg_max = getattr(head, "reg_max", 16)

    @tf.function()
    def _forward(self, images: tf.Tensor):
        out = self.model(images, training=False)
        cls_list = out["cls"]
        reg_list = out["reg"]
        obj_list = out.get("obj", [tf.zeros_like(cls_list[0])[..., :1]] * len(cls_list))
        grids = out["grids"]
        strides = out["strides"]

        img_h = tf.cast(tf.shape(images)[1], tf.float32)
        img_w = tf.cast(tf.shape(images)[2], tf.float32)

        boxes_all = []
        scores_all = []
        for cls_map, reg_map, obj_map, pts, stride in zip(cls_list, reg_list, obj_list, grids, strides):
            dist = integral_distribution(reg_map, reg_max=self.reg_max) * float(stride)
            pts_b = tf.tile(pts[None, ...], [tf.shape(images)[0], 1, 1])
            x1 = pts_b[..., 0] - dist[..., 0]
            y1 = pts_b[..., 1] - dist[..., 1]
            x2 = pts_b[..., 0] + dist[..., 2]
            y2 = pts_b[..., 1] + dist[..., 3]
            # Normalize and convert to [y1, x1, y2, x2] for combined_nms
            y1n = tf.clip_by_value(y1 / img_h, 0.0, 1.0)
            x1n = tf.clip_by_value(x1 / img_w, 0.0, 1.0)
            y2n = tf.clip_by_value(y2 / img_h, 0.0, 1.0)
            x2n = tf.clip_by_value(x2 / img_w, 0.0, 1.0)
            boxes = tf.stack([y1n, x1n, y2n, x2n], axis=-1)
            boxes_all.append(boxes)
            scores_all.append(tf.sigmoid(cls_map) * tf.sigmoid(obj_map))

        boxes = tf.concat(boxes_all, axis=1)  # [B, N, 4] in [y1,x1,y2,x2]
        scores = tf.concat(scores_all, axis=1)  # [B, N, C]
        boxes, scores, classes, valid = combined_nms(
            boxes, scores, score_thresh=self.score_thresh, iou_thresh=self.iou_thresh
        )
        # Reorder boxes back to [x1, y1, x2, y2]
        boxes_xyxy = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)
        return boxes_xyxy, scores, classes, valid

    def predict(self, images: tf.Tensor):
        return self._forward(images)
