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

    def _decode_outputs(self, cls_list, reg_list, grids, strides, img_h, img_w):
        boxes_all: List[tf.Tensor] = []
        scores_all: List[tf.Tensor] = []
        batch = tf.shape(cls_list[0])[0]

        for cls_map, reg_map, pts, stride in zip(cls_list, reg_list, grids, strides):
            stride = tf.cast(stride, tf.float32)
            dist = integral_distribution(reg_map, reg_max=self.reg_max) * stride
            pts = tf.cast(pts, tf.float32)
            pts = tf.expand_dims(pts, axis=0)
            pts_b = tf.tile(pts, [batch, 1, 1])
            x1 = pts_b[..., 0] - dist[..., 0]
            y1 = pts_b[..., 1] - dist[..., 1]
            x2 = pts_b[..., 0] + dist[..., 2]
            y2 = pts_b[..., 1] + dist[..., 3]

            img_h_f = tf.cast(img_h, tf.float32)
            img_w_f = tf.cast(img_w, tf.float32)
            y1n = tf.clip_by_value(y1 / img_h_f, 0.0, 1.0)
            x1n = tf.clip_by_value(x1 / img_w_f, 0.0, 1.0)
            y2n = tf.clip_by_value(y2 / img_h_f, 0.0, 1.0)
            x2n = tf.clip_by_value(x2 / img_w_f, 0.0, 1.0)
            boxes = tf.stack([y1n, x1n, y2n, x2n], axis=-1)
            boxes_all.append(boxes)
            scores_all.append(tf.sigmoid(cls_map))

        boxes = tf.concat(boxes_all, axis=1)
        scores = tf.concat(scores_all, axis=1)
        boxes, scores, classes, valid = combined_nms(
            boxes, scores, score_thresh=self.score_thresh, iou_thresh=self.iou_thresh
        )
        boxes_xyxy = tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)
        return boxes_xyxy, scores, classes, valid

    @tf.function()
    def _forward(self, images: tf.Tensor):
        out = self.model(images, training=False)
        cls_list = out["cls"]
        reg_list = out["reg"]
        grids = out["grids"]
        strides = out["strides"]

        img_h = tf.cast(tf.shape(images)[1], tf.float32)
        img_w = tf.cast(tf.shape(images)[2], tf.float32)

        return self._decode_outputs(cls_list, reg_list, grids, strides, img_h, img_w)

    def predict(self, images: tf.Tensor):
        return self._forward(images)

    def predict_from_outputs(self, outputs: dict, img_h, img_w):
        cls_list = outputs["cls"]
        reg_list = outputs["reg"]
        grids = outputs["grids"]
        strides = outputs["strides"]
        return self._decode_outputs(cls_list, reg_list, grids, strides, img_h, img_w)

