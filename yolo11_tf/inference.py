from typing import List, Tuple

import tensorflow as tf

from .losses import integral_distribution
from .utils import combined_nms


class YoloInferencer:
    def __init__(self, model: tf.keras.Model, score_thresh: float = 0.25, iou_thresh: float = 0.7):
        self.model = model
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh

    @tf.function(jit_compile=False)
    def _forward(self, images: tf.Tensor):
        out = self.model(images, training=False)
        cls_list = out["cls"]
        reg_list = out["reg"]
        grids = out["grids"]
        strides = out["strides"]

        boxes_all = []
        scores_all = []
        for cls_map, reg_map, pts, stride in zip(cls_list, reg_list, grids, strides):
            dist = integral_distribution(reg_map, reg_max=16) * float(stride)
            pts_b = tf.tile(pts[None, ...], [tf.shape(images)[0], 1, 1])
            x1 = pts_b[..., 0] - dist[..., 0]
            y1 = pts_b[..., 1] - dist[..., 1]
            x2 = pts_b[..., 0] + dist[..., 2]
            y2 = pts_b[..., 1] + dist[..., 3]
            boxes = tf.stack([x1, y1, x2, y2], axis=-1)
            boxes_all.append(boxes)
            scores_all.append(tf.sigmoid(cls_map))

        boxes = tf.concat(boxes_all, axis=1)  # [B, N, 4]
        scores = tf.concat(scores_all, axis=1)  # [B, N, C]
        nmsed = combined_nms(boxes, scores, score_thresh=self.score_thresh, iou_thresh=self.iou_thresh)
        return nmsed

    def predict(self, images: tf.Tensor):
        return self._forward(images)

