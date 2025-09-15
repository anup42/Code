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


# ---------- Additional YOLO-style utility losses ----------
def bce_with_logits_loss(logits, labels):
    """Elementwise BCE with logits (no reduction)."""
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)


def integral_distribution(logits, reg_max: int):
    """Convert DFL logits to expected distances via integral over bins.

    logits: [..., 4*(reg_max+1)] or [..., 4, (reg_max+1)]
    Returns: same leading dims with last dim=4 (expected distances per side)
    """
    bins = reg_max + 1
    shape = tf.shape(logits)
    last = logits.shape[-1]
    if last is None:
        last = shape[-1]
    # If last dim is bins, assume shape [..., 4, bins]
    if last == bins:
        x = logits
    elif last == 4 * bins:
        x = tf.reshape(logits, tf.concat([shape[:-1], tf.constant([4, bins])], axis=0))
    else:
        raise ValueError("integral_distribution expects last dim == bins or 4*bins")
    prob = tf.nn.softmax(x, axis=-1)
    idx = tf.cast(tf.range(bins)[None, :], tf.float32)  # [1, bins]
    exp = tf.reduce_sum(prob * idx, axis=-1)  # [..., 4]
    return exp


def dfl_loss(logits, target, reg_max: int):
    """Distribution Focal Loss for bounding box regression.

    logits: [M, 4, bins]
    target: [M, 4] in [0, reg_max]
    Returns: scalar loss
    """
    bins = reg_max + 1
    target = tf.stop_gradient(tf.clip_by_value(target, 0.0, float(reg_max)))
    tl = tf.floor(target)
    tr = tf.minimum(tl + 1.0, float(reg_max))
    wl = tr - target
    wr = target - tl

    tl = tf.cast(tl, tf.int32)
    tr = tf.cast(tr, tf.int32)

    log_prob = tf.nn.log_softmax(logits, axis=-1)  # [M,4,bins]
    onehot_l = tf.one_hot(tl, depth=bins, dtype=log_prob.dtype)  # [M,4,bins]
    onehot_r = tf.one_hot(tr, depth=bins, dtype=log_prob.dtype)
    ce_l = -tf.reduce_sum(onehot_l * log_prob, axis=-1)  # [M,4]
    ce_r = -tf.reduce_sum(onehot_r * log_prob, axis=-1)
    loss = wl * ce_l + wr * ce_r  # [M,4]
    return tf.reduce_mean(loss)


def bbox_ciou(pred_boxes, gt_boxes, eps=1e-7):
    """CIoU loss: returns (1 - ciou).

    boxes: [N,4] xyxy
    """
    # IoU
    x1 = tf.maximum(pred_boxes[..., 0], gt_boxes[..., 0])
    y1 = tf.maximum(pred_boxes[..., 1], gt_boxes[..., 1])
    x2 = tf.minimum(pred_boxes[..., 2], gt_boxes[..., 2])
    y2 = tf.minimum(pred_boxes[..., 3], gt_boxes[..., 3])
    inter = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    area1 = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    area2 = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])
    union = area1 + area2 - inter + eps
    iou = inter / union

    # center distance
    px = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2.0
    py = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2.0
    gx = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2.0
    gy = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2.0
    center_dist = (px - gx) ** 2 + (py - gy) ** 2

    # enclosing box diagonal squared
    cx1 = tf.minimum(pred_boxes[..., 0], gt_boxes[..., 0])
    cy1 = tf.minimum(pred_boxes[..., 1], gt_boxes[..., 1])
    cx2 = tf.maximum(pred_boxes[..., 2], gt_boxes[..., 2])
    cy2 = tf.maximum(pred_boxes[..., 3], gt_boxes[..., 3])
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps

    # aspect ratio consistency term v
    w1 = tf.maximum(pred_boxes[..., 2] - pred_boxes[..., 0], 0.0)
    h1 = tf.maximum(pred_boxes[..., 3] - pred_boxes[..., 1], 0.0)
    w2 = tf.maximum(gt_boxes[..., 2] - gt_boxes[..., 0], 0.0)
    h2 = tf.maximum(gt_boxes[..., 3] - gt_boxes[..., 1], 0.0)
    v = (4 / (3.14159265 ** 2)) * tf.pow(tf.atan(w2 / (h2 + eps)) - tf.atan(w1 / (h1 + eps)), 2)
    alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist / c2) - alpha * v
    return 1.0 - ciou


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
