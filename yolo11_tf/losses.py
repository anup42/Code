import tensorflow as tf


def bbox_iou(box1, box2, eps=1e-7):
    # box: [x1, y1, x2, y2]
    inter_x1 = tf.maximum(box1[..., 0], box2[..., 0])
    inter_y1 = tf.maximum(box1[..., 1], box2[..., 1])
    inter_x2 = tf.minimum(box1[..., 2], box2[..., 2])
    inter_y2 = tf.minimum(box1[..., 3], box2[..., 3])
    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - inter + eps
    return inter / union


def xywh_to_xyxy(xywh):
    cx, cy, w, h = tf.split(xywh, 4, axis=-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return tf.concat([x1, y1, x2, y2], axis=-1)


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, box_weight=7.5, obj_weight=1.0, cls_weight=0.5, name="yolo_loss"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight

    def call(self, y_true, y_pred):
        # y_true, y_pred are lists of 3 tensors per scale
        total_box, total_obj, total_cls = 0.0, 0.0, 0.0
        for t, p in zip(y_true, y_pred):
            # p: [B, H, W, 5 + C] => [box(4), obj(1), cls(C)]
            pred_box = p[..., 0:4]
            pred_obj = p[..., 4:5]
            pred_cls = p[..., 5:]

            tgt_box = t[..., 0:4]
            tgt_obj = t[..., 4:5]
            tgt_cls = t[..., 5:]

            # IoU loss on positives
            iou = bbox_iou(xywh_to_xyxy(pred_box), xywh_to_xyxy(tgt_box))
            # only where object
            obj_mask = tf.cast(tf.equal(tgt_obj, 1.0), tf.float32)
            iou_loss = (1.0 - iou) * tf.squeeze(obj_mask, axis=-1)
            box_loss = tf.reduce_sum(iou_loss) / (tf.reduce_sum(obj_mask) + 1e-6)

            # Objectness loss
            bce = tf.keras.losses.binary_crossentropy(tgt_obj, tf.sigmoid(pred_obj), from_logits=False)
            obj_loss = tf.reduce_mean(bce)

            # Classification loss on positives only
            if self.num_classes > 1:
                # Use per-element sigmoid CE to avoid implicit reductions
                cls_loss_map = tf.nn.sigmoid_cross_entropy_with_logits(labels=tgt_cls, logits=pred_cls)
                cls_loss = tf.reduce_sum(cls_loss_map * obj_mask)
                cls_loss = cls_loss / (tf.reduce_sum(obj_mask) + 1e-6)
            else:
                cls_loss = 0.0

            total_box += box_loss
            total_obj += obj_loss
            total_cls += cls_loss

        return self.box_weight * total_box + self.obj_weight * total_obj + self.cls_weight * total_cls
