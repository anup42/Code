from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from .losses import bce_with_logits_loss, bbox_ciou, integral_distribution



def dist2bbox_tf(distance: tf.Tensor, anchor_points: tf.Tensor, xywh: bool = False) -> tf.Tensor:
    distance = tf.convert_to_tensor(distance, dtype=tf.float32)
    anchor_points = tf.convert_to_tensor(anchor_points, dtype=tf.float32)
    anchor = anchor_points
    if tf.rank(anchor_points) == 2:
        anchor = tf.expand_dims(anchor_points, axis=0)
    while tf.rank(anchor) < tf.rank(distance):
        anchor = tf.expand_dims(anchor, axis=0)
    lt, rb = tf.split(distance, 2, axis=-1)
    anchor = tf.cast(anchor, distance.dtype)
    x1y1 = anchor - lt
    x2y2 = anchor + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2.0
        wh = x2y2 - x1y1
        return tf.concat([c_xy, wh], axis=-1)
    return tf.concat([x1y1, x2y2], axis=-1)


def bbox2dist_tf(anchor_points: tf.Tensor, bboxes: tf.Tensor, reg_max: int) -> tf.Tensor:
    anchor_points = tf.convert_to_tensor(anchor_points, dtype=tf.float32)
    bboxes = tf.convert_to_tensor(bboxes, dtype=tf.float32)
    anchor = anchor_points
    if tf.rank(anchor_points) == 2:
        anchor = tf.expand_dims(anchor_points, axis=0)
    while tf.rank(anchor) < tf.rank(bboxes):
        anchor = tf.expand_dims(anchor, axis=0)
    x1y1, x2y2 = tf.split(bboxes, 2, axis=-1)
    anchor = tf.cast(anchor, bboxes.dtype)
    dists = tf.concat([anchor - x1y1, x2y2 - anchor], axis=-1)
    return tf.clip_by_value(dists, 0.0, float(reg_max) - 0.01)


def bbox_iou_ciou_tf(pred_boxes: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    pred_boxes = tf.convert_to_tensor(pred_boxes, dtype=tf.float32)
    gt_boxes = tf.convert_to_tensor(gt_boxes, dtype=tf.float32)
    ciou_loss = bbox_ciou(pred_boxes, gt_boxes)
    return tf.maximum(1.0 - ciou_loss, 0.0)


def distribution_focal_loss_tf(logits: tf.Tensor, target: tf.Tensor, reg_max: int) -> tf.Tensor:
    bins = reg_max + 1
    target = tf.clip_by_value(target, 0.0, float(reg_max))
    tl = tf.floor(target)
    tr = tf.minimum(tl + 1.0, float(reg_max))
    wl = tr - target
    wr = target - tl
    tl = tf.cast(tl, tf.int32)
    tr = tf.cast(tr, tf.int32)
    log_prob = tf.nn.log_softmax(logits, axis=-1)
    onehot_l = tf.one_hot(tl, depth=bins, dtype=log_prob.dtype)
    onehot_r = tf.one_hot(tr, depth=bins, dtype=log_prob.dtype)
    ce_l = -tf.reduce_sum(onehot_l * log_prob, axis=-1)
    ce_r = -tf.reduce_sum(onehot_r * log_prob, axis=-1)
    return wl * ce_l + wr * ce_r


class TaskAlignedAssignerTF:
    def __init__(self, topk: int = 10, num_classes: int = 80, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9):
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def __call__(
        self,
        pd_scores: tf.Tensor,
        pd_bboxes: tf.Tensor,
        anc_points: tf.Tensor,
        gt_labels: tf.Tensor,
        gt_bboxes: tf.Tensor,
        mask_gt: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        bs = tf.shape(pd_scores)[0]
        n_max_boxes = tf.shape(gt_bboxes)[1]
        num_anchors = tf.shape(pd_scores)[1]

        def no_targets():
            target_labels = tf.fill([bs, num_anchors], tf.cast(self.num_classes, tf.int32))
            target_bboxes = tf.zeros([bs, num_anchors, 4], dtype=tf.float32)
            target_scores = tf.zeros([bs, num_anchors, self.num_classes], dtype=tf.float32)
            fg_mask = tf.zeros([bs, num_anchors], dtype=tf.bool)
            target_gt_idx = tf.zeros([bs, num_anchors], dtype=tf.int32)
            return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

        def assign():
            mask_candidates = self.select_candidates_in_gts(anc_points, gt_bboxes)
            mask_gt_bool = tf.cast(mask_gt, tf.bool)
            mask_gt_full = tf.broadcast_to(mask_gt_bool, tf.shape(mask_candidates))
            align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_candidates & mask_gt_full)
            topk_mask = tf.tile(mask_gt_bool, [1, 1, self.topk])
            mask_topk = self.select_topk_candidates(align_metric, topk_mask=topk_mask)
            mask_pos = mask_topk * tf.cast(mask_candidates, mask_topk.dtype) * tf.cast(mask_gt_full, mask_topk.dtype)
            target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps)
            target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
            align_metric = align_metric * mask_pos
            pos_align_metrics = tf.reduce_max(align_metric, axis=-1, keepdims=True)
            pos_overlaps = tf.reduce_max(overlaps * mask_pos, axis=-1, keepdims=True)
            norm_align_metric = tf.reduce_max(
                align_metric * pos_overlaps / (pos_align_metrics + self.eps),
                axis=-2,
                keepdims=True,
            )
            target_scores = target_scores * norm_align_metric
            return target_labels, target_bboxes, target_scores, tf.cast(fg_mask > 0, tf.bool), target_gt_idx

        return tf.cond(tf.equal(n_max_boxes, 0), no_targets, assign)

    def select_candidates_in_gts(self, anc_points: tf.Tensor, gt_bboxes: tf.Tensor) -> tf.Tensor:
        anc_points = tf.cast(anc_points, tf.float32)
        gt_bboxes = tf.cast(gt_bboxes, tf.float32)
        anc = tf.reshape(anc_points, [1, 1, -1, 2])
        x1y1 = gt_bboxes[..., :2][..., None, :]
        x2y2 = gt_bboxes[..., 2:][..., None, :]
        deltas = tf.concat([anc - x1y1, x2y2 - anc], axis=-1)
        return tf.reduce_min(deltas, axis=-1) > 0.0

    def get_box_metrics(
        self,
        pd_scores: tf.Tensor,
        pd_bboxes: tf.Tensor,
        gt_labels: tf.Tensor,
        gt_bboxes: tf.Tensor,
        mask_gt: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        mask_gt_bool = tf.cast(mask_gt, tf.bool)
        bs = tf.shape(pd_scores)[0]
        n_max_boxes = tf.shape(gt_bboxes)[1]
        num_anchors = tf.shape(pd_scores)[1]
        gt_labels_squeezed = tf.squeeze(tf.cast(gt_labels, tf.int32), axis=-1)
        one_hot = tf.one_hot(gt_labels_squeezed, depth=self.num_classes, dtype=pd_scores.dtype)
        score_per_gt = tf.einsum('bmc,bnc->bmn', one_hot, pd_scores)
        bbox_scores = tf.where(mask_gt_bool, score_per_gt, tf.zeros_like(score_per_gt))
        pred_boxes = tf.expand_dims(pd_bboxes, axis=1)
        gt_boxes = tf.expand_dims(gt_bboxes, axis=2)
        overlaps = bbox_iou_ciou_tf(pred_boxes, gt_boxes)
        overlaps = tf.where(mask_gt_bool, overlaps, tf.zeros_like(overlaps))
        align_metric = tf.pow(bbox_scores, self.alpha) * tf.pow(overlaps, self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics: tf.Tensor, topk_mask: tf.Tensor | None = None) -> tf.Tensor:
        k = self.topk
        topk_vals, topk_idx = tf.math.top_k(metrics, k=k, sorted=False)
        if topk_mask is None:
            topk_valid = tf.reduce_max(topk_vals, axis=-1, keepdims=True) > self.eps
            topk_mask = tf.tile(topk_valid, [1, 1, k])
        depth = tf.shape(metrics)[-1]
        one_hot = tf.one_hot(topk_idx, depth=depth, dtype=tf.int32)
        one_hot = one_hot * tf.cast(topk_mask[..., None], tf.int32)
        count_tensor = tf.reduce_sum(one_hot, axis=-2)
        count_tensor = tf.where(count_tensor > 1, 0, count_tensor)
        return tf.cast(count_tensor, metrics.dtype)

    def select_highest_overlaps(self, mask_pos: tf.Tensor, overlaps: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        fg_mask = tf.reduce_sum(mask_pos, axis=-2)
        mask_multi = fg_mask > 1
        n_boxes = tf.shape(mask_pos)[1]
        if tf.reduce_any(mask_multi):
            mask_multi_gts = tf.tile(mask_multi[:, tf.newaxis, :], [1, n_boxes, 1])
            max_overlaps_idx = tf.argmax(overlaps, axis=1)
            is_max = tf.one_hot(max_overlaps_idx, depth=n_boxes, axis=1, dtype=mask_pos.dtype)
            mask_pos = tf.where(mask_multi_gts, is_max, mask_pos)
            fg_mask = tf.reduce_sum(mask_pos, axis=-2)
        target_gt_idx = tf.argmax(mask_pos, axis=-2, output_type=tf.int32)
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(
        self,
        gt_labels: tf.Tensor,
        gt_bboxes: tf.Tensor,
        target_gt_idx: tf.Tensor,
        fg_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        bs = tf.shape(gt_labels)[0]
        n_max_boxes = tf.shape(gt_labels)[1]
        num_anchors = tf.shape(target_gt_idx)[1]
        batch_ind = tf.range(bs, dtype=tf.int32)[:, None]
        flat_gt_idx = target_gt_idx + batch_ind * n_max_boxes
        flat_labels = tf.reshape(tf.cast(gt_labels, tf.int32), [-1])
        target_labels = tf.gather(flat_labels, flat_gt_idx)
        flat_bboxes = tf.reshape(tf.cast(gt_bboxes, tf.float32), [-1, 4])
        target_bboxes = tf.gather(flat_bboxes, flat_gt_idx)
        target_labels = tf.where(fg_mask > 0, target_labels, tf.zeros_like(target_labels))
        max_label = tf.cast(self.num_classes - 1, target_labels.dtype)
        target_labels = tf.clip_by_value(target_labels, 0, max_label)
        target_scores = tf.one_hot(target_labels, depth=self.num_classes, dtype=tf.float32)
        target_scores = tf.where(fg_mask[..., None] > 0, target_scores, tf.zeros_like(target_scores))
        return target_labels, target_bboxes, target_scores

@dataclass
class TrainConfig:
    num_classes: int
    img_size: int = 640
    reg_max: int = 16
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 100
    warmup_epochs: int = 3
    cls_loss_gain: float = 1.0
    box_loss_gain: float = 7.5
    dfl_loss_gain: float = 1.5
    max_boxes: int = 300
    # Performance/stability knobs
    prefer_gpu_ops: bool = True
    debug_asserts: bool = False
    cr_topk_limit: int = 4096


class Trainer:
    def __init__(
        self,
        model: tf.keras.Model,
        cfg: TrainConfig,
        strategy: Optional[tf.distribute.Strategy] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.strategy = strategy or tf.distribute.get_strategy()
        self._num_replicas = max(1, int(getattr(self.strategy, "num_replicas_in_sync", 1)))
        if cfg.weight_decay > 0:
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
        self.assigner = TaskAlignedAssignerTF(topk=10, num_classes=cfg.num_classes)
        # Training state for checkpointing/resume support
        self._epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name="epoch")
        self._global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
        self.ckpt = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self._epoch,
            global_step=self._global_step,
        )

    def assign_epoch(self, epoch: int):
        """Update the stored epoch counter (number of completed epochs)."""
        self._epoch.assign(int(epoch))

    def current_epoch(self) -> int:
        """Return the stored epoch counter as a Python int."""
        return int(self._epoch.numpy())

    def global_step_value(self) -> int:
        """Return the global step counter as a Python int."""
        return int(self._global_step.numpy())

    def restore(self, checkpoint_path: str, expect_partial: bool = True):
        """Restore model/optimizer state from a checkpoint file."""
        status = self.ckpt.restore(checkpoint_path)
        if expect_partial:
            status.expect_partial()
        else:
            status.assert_existing_objects_matched()
        return status

    def _gather_cpu(self, x, indices, batch_dims=0):
        if getattr(self.cfg, 'prefer_gpu_ops', True):
            return tf.gather(x, indices, batch_dims=batch_dims)
        with tf.device('/CPU:0'):
            return tf.gather(x, indices, batch_dims=batch_dims)

    def _gather_nd_cpu(self, x, indices):
        if getattr(self.cfg, 'prefer_gpu_ops', True):
            return tf.gather_nd(x, indices)
        with tf.device('/CPU:0'):
            return tf.gather_nd(x, indices)

    def _gather_hw(self, tensor, batch_indices, hw_indices, name="hw_idx"):
        """Gather values from [B, HW, ...] tensors using per-element batch/HW indices.

        This avoids two-step gathers (first by batch then by HW) whose gradients can
        trigger very large `UnsortedSegment` launches on GPU. The combined gather_nd
        keeps the inner dimension small, preventing overflow in the CUDA kernel.
        """
        batch_indices = tf.cast(batch_indices, tf.int32)
        hw_indices = tf.cast(hw_indices, tf.int32)
        if getattr(self.cfg, 'debug_asserts', False):
            batch_indices = self._assert_indices_in_range(
                batch_indices, tf.shape(tensor)[0], name=f"{name}_batch"
            )
            hw_indices = self._assert_indices_in_range(
                hw_indices, tf.shape(tensor)[1], name=name
            )
        gather_idx = tf.stack([batch_indices, hw_indices], axis=1)
        if getattr(self.cfg, 'prefer_gpu_ops', True):
            return tf.gather_nd(tensor, gather_idx)
        with tf.device('/CPU:0'):
            return tf.gather_nd(tensor, gather_idx)

    def _top_k(self, values, k):
        """Top-k with optional CPU placement for stability when prefer_gpu_ops=False."""
        if getattr(self.cfg, 'prefer_gpu_ops', True):
            return tf.math.top_k(values, k=k)
        with tf.device('/CPU:0'):
            return tf.math.top_k(values, k=k)

    @staticmethod
    def _assert_indices_in_range(idx: tf.Tensor, upper: tf.Tensor, name: str = "idx"):
        """Assert idx in [0, upper). Returns identity of idx with control deps.
        Uses max/min on augmented tensors to avoid shape-dependent control flow.
        """
        idx = tf.convert_to_tensor(idx)
        upper = tf.cast(upper, idx.dtype)
        flat = tf.reshape(idx, [-1])
        # If idx is empty, safe_min=0, safe_max=-1
        safe_min = tf.reduce_min(tf.concat([flat, tf.zeros([1], dtype=idx.dtype)], axis=0))
        safe_max = tf.reduce_max(tf.concat([flat, tf.constant([-1], dtype=idx.dtype)], axis=0))
        with tf.control_dependencies([
            tf.debugging.assert_greater_equal(safe_min, tf.cast(0, idx.dtype), message=f"{name} has negative values"),
            tf.debugging.assert_less(safe_max, upper, message=f"{name} contains values >= upper bound"),
        ]):
            return tf.identity(idx)

    @staticmethod
    def _build_grids_from_outputs(outputs: Dict):
        # Already provided per forward pass
        return outputs["grids"], outputs["strides"]

    def _compute_loss_components(self, outputs, targets):
        cls_outs: List[tf.Tensor] = outputs["cls"]
        reg_outs: List[tf.Tensor] = outputs["reg"]
        grids = outputs["grids"]
        strides = outputs["strides"]

        bins = self.cfg.reg_max + 1

        cls_list = []
        reg_list = []
        anchor_points_grid = []
        stride_list = []
        for i, (cls_map, reg_map, grid) in enumerate(zip(cls_outs, reg_outs, grids)):
            cls_map = tf.cast(cls_map, tf.float32)
            reg_map = tf.cast(reg_map, tf.float32)
            grid = tf.cast(grid, tf.float32)
            stride = tf.cast(strides[i], tf.float32)
            cls_list.append(cls_map)
            reg_list.append(reg_map)
            anchor_points_grid.append(grid / stride)
            stride_list.append(tf.ones([tf.shape(grid)[0], 1], dtype=tf.float32) * stride)

        pred_scores = tf.concat(cls_list, axis=1)
        reg_pred = tf.concat(reg_list, axis=1)
        anchor_points_grid = tf.concat(anchor_points_grid, axis=0)
        stride_tensor = tf.concat(stride_list, axis=0)
        anchor_points_pixel = anchor_points_grid * stride_tensor

        pred_scores_prob = tf.nn.sigmoid(pred_scores)
        pred_dist_expect = integral_distribution(reg_pred, self.cfg.reg_max)
        pred_bboxes_grid = dist2bbox_tf(pred_dist_expect, anchor_points_grid)
        pred_bboxes = pred_bboxes_grid * stride_tensor[None, :, :]

        gt_labels = tf.cast(targets[..., 0:1], tf.int32)
        gt_bboxes = tf.cast(targets[..., 1:5], tf.float32)
        mask_gt = tf.cast(targets[..., 5:6] > 0.5, tf.bool)

        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores_prob,
            pred_bboxes,
            anchor_points_pixel,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_bboxes_grid = target_bboxes / stride_tensor[None, :, :]
        target_scores_sum = tf.maximum(tf.reduce_sum(target_scores), 1.0)

        cls_loss_map = bce_with_logits_loss(pred_scores, target_scores)
        total_cls = tf.reduce_sum(cls_loss_map) / target_scores_sum

        weight = tf.reduce_sum(target_scores, axis=-1)
        fg_mask_bool = fg_mask
        indices = tf.where(fg_mask_bool)
        total_pos = tf.reduce_sum(tf.cast(fg_mask_bool, tf.float32))

        pred_dist_logits = tf.reshape(reg_pred, [tf.shape(reg_pred)[0], -1, 4, bins])
        target_ltrb = bbox2dist_tf(anchor_points_grid, target_bboxes_grid, self.cfg.reg_max)

        def no_pos():
            return tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)

        def pos_losses():
            weight_fg = tf.gather_nd(weight, indices)
            pred_boxes_fg = tf.gather_nd(pred_bboxes_grid, indices)
            target_boxes_fg = tf.gather_nd(target_bboxes_grid, indices)
            ciou_loss = bbox_ciou(pred_boxes_fg, target_boxes_fg)
            loss_iou = tf.reduce_sum(ciou_loss * weight_fg) / target_scores_sum
            pred_dist_fg = tf.gather_nd(pred_dist_logits, indices)
            target_ltrb_fg = tf.gather_nd(target_ltrb, indices)
            dfl_per = distribution_focal_loss_tf(pred_dist_fg, target_ltrb_fg, self.cfg.reg_max)
            loss_dfl = tf.reduce_sum(dfl_per * tf.expand_dims(weight_fg, -1)) / target_scores_sum
            return loss_iou, loss_dfl

        total_box, total_dfl = tf.cond(tf.shape(indices)[0] > 0, pos_losses, no_pos)

        loss = (
            self.cfg.box_loss_gain * total_box
            + self.cfg.cls_loss_gain * total_cls
            + self.cfg.dfl_loss_gain * total_dfl
        )

        metrics = {
            "loss": tf.cast(loss, tf.float32),
            "cls": tf.cast(total_cls, tf.float32),
            "box": tf.cast(total_box, tf.float32),
            "dfl": tf.cast(total_dfl, tf.float32),
            "pos": tf.cast(total_pos, tf.float32),
            "obj": tf.constant(0.0, tf.float32),
        }
        return loss, metrics

    def distribute_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        if self._num_replicas > 1:
            return self.strategy.experimental_distribute_dataset(dataset)
        return dataset

    def batch_size_from_dataset_elem(self, batch) -> int:
        try:
            images, _ = self.extract_images_targets(batch)
        except ValueError:
            images = batch
        if self._num_replicas > 1 and isinstance(images, tf.distribute.DistributedValues):
            elems = self.strategy.experimental_local_results(images)
            total = 0
            for elem in elems:
                total += int(tf.shape(elem)[0].numpy())
            return total
        return int(tf.shape(images)[0].numpy())

    def _reduce_metrics(self, metrics: Dict[str, tf.Tensor]) -> Dict[str, float]:
        reduced = {}
        for key, value in metrics.items():
            v = value
            if self._num_replicas > 1 and isinstance(value, tf.distribute.DistributedValues):
                v = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
            if isinstance(v, tf.Tensor):
                reduced[key] = float(v.numpy())
            else:
                reduced[key] = float(v)
        return reduced

    def run_train_step(self, batch) -> Dict[str, float]:
        images, targets = self.extract_images_targets(batch)
        if self._num_replicas > 1 and isinstance(images, tf.distribute.DistributedValues):
            per_replica_metrics = self.strategy.run(self.train_step, args=(images, targets))
            metrics = self._reduce_metrics(per_replica_metrics)
            self._global_step.assign_add(1)
            return metrics
        metrics = self.train_step(images, targets)
        out = {}
        for key, value in metrics.items():
            out[key] = float(value.numpy()) if isinstance(value, tf.Tensor) else float(value)
        return out

    def compute_loss_metrics(self, outputs, targets) -> Dict[str, tf.Tensor]:
        loss, metrics = self._compute_loss_components(outputs, targets)
        metrics["loss"] = tf.cast(loss, tf.float32)
        return metrics

    @tf.function(jit_compile=False, experimental_relax_shapes=True)
    def train_step(self, images, targets):
        with tf.GradientTape() as tape:
            outputs = self.model(images, training=True)
            loss, metrics = self._compute_loss_components(outputs, targets)

        loss_for_grad = loss
        if self._num_replicas > 1:
            loss_for_grad = loss / tf.cast(self._num_replicas, loss.dtype)

        grads = tape.gradient(loss_for_grad, self.model.trainable_variables)
        if grads is None:
            grads = []

        all_none = True
        processed_grads = []
        for grad, var in zip(grads, self.model.trainable_variables):
            if grad is None:
                processed_grads.append(tf.zeros_like(var))
            else:
                processed_grads.append(grad)
                all_none = False

        if all_none:
            return metrics

        self.optimizer.apply_gradients(zip(processed_grads, self.model.trainable_variables))

        if self._num_replicas == 1:
            self._global_step.assign_add(1)

        return metrics

    @tf.function(jit_compile=False, experimental_relax_shapes=True)
    def validation_step(self, images, targets):
        outputs = self.model(images, training=False)
        _, metrics = self._compute_loss_components(outputs, targets)
        return metrics

    def extract_images_targets(self, batch):
        images, targets = self._extract_images_targets(batch)
        if images is None or targets is None:
            raise ValueError("Batch does not contain both images and targets")
        return images, targets

    def _extract_images_targets(self, batch):
        if isinstance(batch, dict):
            images = None
            targets = None
            for key in ("images", "image", "x"):
                if key in batch:
                    images = batch[key]
                    break
            for key in ("targets", "target", "labels", "y"):
                if key in batch:
                    targets = batch[key]
                    break
            return images, targets
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 2:
                first, second = batch[0], batch[1]
                if isinstance(first, (tuple, list)):
                    if len(first) >= 2:
                        return first[0], first[1]
                    return self._extract_images_targets(first)
                return first, second
            if len(batch) == 1:
                return self._extract_images_targets(batch[0])
            return None, None
        return batch, None


# =====================
# Lightweight eval utils
# =====================

def _xywhn_to_xyxy_np(xywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def _bbox_iou_np(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    # boxes: [N,4] and [M,4] in xyxy normalized
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    w = np.clip(x2 - x1, 0.0, 1.0)
    h = np.clip(y2 - y1, 0.0, 1.0)
    inter = w * h
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter + eps
    return inter / union


def _nms_np(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> np.ndarray:
    # boxes: [N,4], scores: [N]
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < max_det:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = _bbox_iou_np(boxes[i:i+1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious <= iou_thres]
    return np.array(keep, dtype=np.int32)


def decode_maps_to_dets_np(pred_maps: List[np.ndarray], conf_thres=0.25, iou_thres=0.5, max_det=300) -> List[List[np.ndarray]]:
    """Decode model head outputs [B,H,W,5+C] into per-image detections.
    Returns a list length B, each with an array [K,6] of [x1,y1,x2,y2,score,cls], all normalized.
    """
    B = pred_maps[0].shape[0]
    C = pred_maps[0].shape[-1] - 5
    outs = [[] for _ in range(B)]
    # Concatenate scales
    all_boxes = []
    all_obj = []
    all_cls = []
    for p in pred_maps:
        # p: [B,H,W,5+C]
        H, W = p.shape[1], p.shape[2]
        p = p.reshape(B, H * W, 5 + C)
        box_logits = p[..., 0:4]
        obj_logits = p[..., 4]
        cls_logits = p[..., 5:]
        # Convert to normalized
        box_sig = 1.0 / (1.0 + np.exp(-box_logits))
        obj = 1.0 / (1.0 + np.exp(-obj_logits))
        cls = 1.0 / (1.0 + np.exp(-cls_logits))
        all_boxes.append(box_sig)
        all_obj.append(obj)
        all_cls.append(cls)
    boxes = np.concatenate(all_boxes, axis=1)  # [B,N,4]
    obj = np.concatenate(all_obj, axis=1)      # [B,N]
    cls = np.concatenate(all_cls, axis=1)      # [B,N,C]

    for b in range(B):
        bx = boxes[b]
        ob = obj[b]
        cl = cls[b]
        # build detections per class
        dets_b = []
        xyxy = _xywhn_to_xyxy_np(bx)
        for c in range(C):
            scores = ob * cl[:, c]
            mask = scores >= conf_thres
            if not np.any(mask):
                continue
            b_c = xyxy[mask]
            s_c = scores[mask]
            keep = _nms_np(b_c, s_c, iou_thres, max_det)
            if keep.size > 0:
                kept = np.concatenate([b_c[keep], s_c[keep, None], np.full((keep.size, 1), c, dtype=np.float32)], axis=1)
                dets_b.append(kept)
        if dets_b:
            outs[b] = [np.vstack(dets_b)]
        else:
            outs[b] = [np.zeros((0, 6), dtype=np.float32)]
    # unwrap inner list level
    return [o[0] for o in outs]


def compute_pr_map50(dets_by_img: List[np.ndarray], gts_by_img: List[np.ndarray], num_classes: int, iou_thres=0.5):
    """Compute global precision/recall and mAP@0.5 across all images.
    gts_by_img: each [Ng,5] with [cls, cx, cy, w, h] normalized.
    dets_by_img: each [Nd,6] with [x1,y1,x2,y2,score,cls].
    """
    tp_scores = {c: [] for c in range(num_classes)}
    fp_scores = {c: [] for c in range(num_classes)}
    npos = {c: 0 for c in range(num_classes)}

    for g, d in zip(gts_by_img, dets_by_img):
        if g.size > 0:
            g_cls = g[:, 0].astype(np.int32)
            g_xyxy = _xywhn_to_xyxy_np(g[:, 1:5])
        else:
            g_cls = np.zeros((0,), dtype=np.int32)
            g_xyxy = np.zeros((0, 4), dtype=np.float32)
        if d.size > 0:
            d_xyxy = d[:, 0:4]
            d_scores = d[:, 4]
            d_cls = d[:, 5].astype(np.int32)
        else:
            d_xyxy = np.zeros((0, 4), dtype=np.float32)
            d_scores = np.zeros((0,), dtype=np.float32)
            d_cls = np.zeros((0,), dtype=np.int32)

        # per-class matching
        for c in range(num_classes):
            gt_idx = np.where(g_cls == c)[0]
            det_idx = np.where(d_cls == c)[0]
            npos[c] += gt_idx.size
            if det_idx.size == 0:
                continue
            det_order = det_idx[np.argsort(-d_scores[det_idx])]
            matched = np.zeros(gt_idx.size, dtype=bool)
            for di in det_order:
                if gt_idx.size == 0:
                    fp_scores[c].append(d_scores[di])
                    continue
                ious = _bbox_iou_np(d_xyxy[di:di+1], g_xyxy[gt_idx])[0]
                best = np.argmax(ious)
                if ious[best] >= iou_thres and not matched[best]:
                    matched[best] = True
                    tp_scores[c].append(d_scores[di])
                else:
                    fp_scores[c].append(d_scores[di])

    # compute AP per class
    ap_list = []
    total_tp = 0
    total_fp = 0
    total_pos = 0
    for c in range(num_classes):
        pos = npos[c]
        total_pos += pos
        scores = np.array(tp_scores[c] + fp_scores[c], dtype=np.float32)
        labels = np.array([1] * len(tp_scores[c]) + [0] * len(fp_scores[c]), dtype=np.int32)
        if scores.size == 0:
            ap_list.append(0.0)
            continue
        order = np.argsort(-scores)
        labels = labels[order]
        tp_cum = np.cumsum(labels)
        fp_cum = np.cumsum(1 - labels)
        total_tp += int(tp_cum[-1])
        total_fp += int(fp_cum[-1])
        if pos == 0:
            ap_list.append(0.0)
            continue
        recall = tp_cum / float(pos)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        # integrate AP (VOC2010-style continuous)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        ap_list.append(float(ap))

    precision_global = total_tp / max(total_tp + total_fp, 1e-9)
    recall_global = total_tp / max(total_pos, 1e-9)
    map50 = float(np.mean(ap_list)) if len(ap_list) else 0.0
    return float(precision_global), float(recall_global), map50


def evaluate_dataset_map50(model: tf.keras.Model, val_ds, num_classes: int,
                           conf_thres=0.25, iou_thres=0.5, max_det=300):
    """Run a forward pass on val_ds and compute P/R/mAP@0.5.
    Expects each batch element as ((images, labels), targets) where
    labels is [B, N, 5] in [cls, cx, cy, w, h] normalized.
    """
    dets_all: List[np.ndarray] = []
    gts_all: List[np.ndarray] = []
    for batch in val_ds:
        (images, labels), _targets = batch
        preds = model(images, training=False)  # list of 3 tensors [B,H,W,5+C]
        preds_np = [p.numpy() for p in preds]
        dets = decode_maps_to_dets_np(preds_np, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
        labs = labels.numpy()
        # labels is ragged? If padded, keep non-zero rows
        for i in range(labs.shape[0]):
            li = labs[i]
            # keep rows where w>0 and h>0
            mask = (li[:, 3] > 0) & (li[:, 4] > 0)
            gts_all.append(li[mask])
            dets_all.append(dets[i])
    return compute_pr_map50(dets_all, gts_all, num_classes, iou_thres=iou_thres)
