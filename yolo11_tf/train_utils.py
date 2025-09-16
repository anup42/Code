from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from .losses import bce_with_logits_loss, dfl_loss, bbox_ciou, integral_distribution


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
    def __init__(self, model: tf.keras.Model, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

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

    def _assign_targets(self, targets, grids, strides):
        """Simplified one-to-one assignment: for each GT, pick closest point
        on a scale chosen by size thresholds.
        targets: [B, max_boxes, 6] -> [cls, x1,y1,x2,y2, valid]
        Returns per-scale tensors:
          pos_idx: list of [B, K] indices into HW per scale (-1 for none)
          pos_targs: list of dicts with 'cls': [B, K, C], 'ltrb': [B, K, 4], 'boxes': [B, K, 4]
        """
        B = tf.shape(targets)[0]
        maxb = tf.shape(targets)[1]
        C = self.cfg.num_classes
        # size thresholds (on sqrt(area)) with widened ranges for multi-scale assignment
        s_small = 64.0
        s_large = 160.0
        low_margin = 0.75
        high_margin = 1.25

        pos_idx_list = []
        pos_targ_list = []

        for si, (pts, stride) in enumerate(zip(grids, strides)):
            N = tf.shape(pts)[0]
            pts_b = tf.tile(pts[None, ...], [B, 1, 1])  # [B, N, 2]

            # Initialize with -1 indices (no assignment)
            pos_idx = tf.fill([B, maxb], tf.constant(-1, dtype=tf.int32))
            cls_t = tf.zeros([B, maxb, C], dtype=tf.float32)
            ltrb_t = tf.zeros([B, maxb, 4], dtype=tf.float32)
            box_t = tf.zeros([B, maxb, 4], dtype=tf.float32)

            cls_ids = tf.cast(targets[..., 0], tf.int32)
            x1, y1, x2, y2, v = tf.split(targets[..., 1:6], [1, 1, 1, 1, 1], axis=-1)
            w = tf.squeeze(x2 - x1, -1)
            h = tf.squeeze(y2 - y1, -1)
            size = tf.sqrt(tf.maximum(w * h, 0.0))

            # choose scale mask (widened boundaries, allow overlap between adjacent scales)
            if si == 0:
                scale_mask = size < (s_small * high_margin)
            elif si == 1:
                scale_mask = tf.logical_and(size >= (s_small * low_margin), size < (s_large * high_margin))
            else:
                scale_mask = size >= (s_large * low_margin)
            scale_mask = tf.logical_and(scale_mask, tf.squeeze(v > 0.5, -1))

            # centers
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers = tf.concat([cx, cy], axis=-1)  # [B, maxb, 2]

            # Distance from points to centers
            # pts_b: [B, N, 2], centers: [B, maxb, 2]
            d = tf.norm(
                tf.expand_dims(pts_b, axis=2) - tf.expand_dims(centers, axis=1),
                axis=-1,
            )  # [B, N, maxb]

            # Prefer points inside the GT box (center sampling)
            pts_x = tf.expand_dims(pts_b[..., 0], -1)  # [B,N,1]
            pts_y = tf.expand_dims(pts_b[..., 1], -1)  # [B,N,1]
            x1t = tf.transpose(x1, [0, 2, 1])  # [B,1,maxb]
            y1t = tf.transpose(y1, [0, 2, 1])
            x2t = tf.transpose(x2, [0, 2, 1])
            y2t = tf.transpose(y2, [0, 2, 1])
            inside = tf.logical_and(tf.logical_and(pts_x >= x1t, pts_x <= x2t),
                                     tf.logical_and(pts_y >= y1t, pts_y <= y2t))  # [B,N,maxb]
            big = tf.constant(1e9, dtype=d.dtype)
            d_inside = tf.where(inside, d, tf.fill(tf.shape(d), big))
            nearest_inside = tf.argmin(d_inside, axis=1, output_type=tf.int32)  # [B,maxb]
            has_inside = tf.reduce_any(inside, axis=1)  # [B,maxb]
            nearest_any = tf.argmin(d, axis=1, output_type=tf.int32)
            chosen = tf.where(has_inside, nearest_inside, nearest_any)

            # Mask out invalids by setting idx to -1
            pos_idx = tf.where(scale_mask, chosen, tf.fill(tf.shape(chosen), tf.constant(-1, dtype=tf.int32)))

            # Build targets for valid ones
            valid_mask = scale_mask
            b_idx = tf.where(valid_mask)
            b_idx = tf.cast(b_idx, tf.int32)
            if tf.shape(b_idx)[0] > 0:
                # b_idx gives indices into [B, maxb]
                bb = b_idx[:, 0]
                gg = b_idx[:, 1]
                # gather point coords
                pidx = tf.gather_nd(pos_idx, b_idx)  # [M]
                pxy = tf.gather(pts, pidx)
                # gather boxes
                gx1 = tf.gather_nd(tf.squeeze(x1, -1), b_idx)
                gy1 = tf.gather_nd(tf.squeeze(y1, -1), b_idx)
                gx2 = tf.gather_nd(tf.squeeze(x2, -1), b_idx)
                gy2 = tf.gather_nd(tf.squeeze(y2, -1), b_idx)
                boxes = tf.stack([gx1, gy1, gx2, gy2], axis=-1)
                # ltrb distances in stride units
                l = (pxy[:, 0] - gx1) / float(stride)
                t = (pxy[:, 1] - gy1) / float(stride)
                r = (gx2 - pxy[:, 0]) / float(stride)
                b = (gy2 - pxy[:, 1]) / float(stride)
                dists = tf.stack([l, t, r, b], axis=-1)

                # clamp to [0, reg_max]
                dists = tf.clip_by_value(dists, 0.0, float(self.cfg.reg_max))

                # write into tensors
                cls_oh = tf.one_hot(tf.gather_nd(cls_ids, b_idx), depth=self.cfg.num_classes)
                if getattr(self.cfg, 'prefer_gpu_ops', True):
                    cls_t = tf.tensor_scatter_nd_update(cls_t, b_idx, cls_oh)
                    ltrb_t = tf.tensor_scatter_nd_update(ltrb_t, b_idx, dists)
                    box_t = tf.tensor_scatter_nd_update(box_t, b_idx, boxes)
                else:
                    with tf.device('/CPU:0'):
                        cls_t = tf.tensor_scatter_nd_update(cls_t, b_idx, cls_oh)
                        ltrb_t = tf.tensor_scatter_nd_update(ltrb_t, b_idx, dists)
                        box_t = tf.tensor_scatter_nd_update(box_t, b_idx, boxes)

            pos_idx_list.append(pos_idx)
            pos_targ_list.append({"cls": cls_t, "ltrb": ltrb_t, "boxes": box_t})

        return pos_idx_list, pos_targ_list

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, images, targets):
        with tf.GradientTape() as tape:
            outputs = self.model(images, training=True)
            cls_outs: List[tf.Tensor] = outputs["cls"]  # 3 tensors [B, HW, C]
            reg_outs: List[tf.Tensor] = outputs["reg"]  # 3 tensors [B, HW, 4*(R)]
            obj_outs: List[tf.Tensor] = outputs.get("obj", [tf.zeros_like(outputs["cls"][0])[..., :1]] * 3)  # [B,HW,1]
            grids = outputs["grids"]
            strides = outputs["strides"]

            pos_idx, pos_targs = self._assign_targets(targets, grids, strides)

            total_cls = 0.0
            total_box = 0.0
            total_dfl = 0.0
            total_pos = 0.0
            total_obj = 0.0

            decoded_boxes_all = []
            cls_scores_all = []

            for si in range(3):
                cls_map = cls_outs[si]
                reg_map = reg_outs[si]
                obj_map = obj_outs[si]
                pts = grids[si]
                stride = float(strides[si])

                # gather positives per image using indices
                idx = pos_idx[si]  # [B, maxb] indices into HW or -1
                B = tf.shape(cls_map)[0]
                HW = tf.shape(cls_map)[1]
                C = tf.shape(cls_map)[2]

                # Sanity: grid size should match HW
                tf.debugging.assert_equal(tf.shape(pts)[0], HW, message="Grid HW mismatch against head output")

                mask = idx >= 0  # [B, maxb]
                num_pos = tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-9
                total_pos += num_pos

                # For classification: build targets per location (one-hot per GT) squeezed onto unique indices.
                # Simpler: compute classification loss only at positive gt selections.
                b_idx, g_idx = tf.where(mask)[:, 0], tf.where(mask)[:, 1]
                b_idx = tf.cast(b_idx, tf.int32)
                g_idx = tf.cast(g_idx, tf.int32)
                lin_idx = tf.zeros([0], dtype=tf.int32)
                if tf.shape(b_idx)[0] > 0:
                    # Compute linear indices on CPU for stability
                    lin_idx = tf.gather_nd(idx, tf.stack([b_idx, g_idx], axis=1))  # [M]
                    if getattr(self.cfg, 'debug_asserts', False):
                        lin_idx = self._assert_indices_in_range(lin_idx, HW, name="lin_idx")
                    pred_cls_sel = self._gather_cpu(cls_map, b_idx)
                    pred_cls_sel = self._gather_cpu(pred_cls_sel, lin_idx, batch_dims=1)  # [M, C]

                    tgt_cls = self._gather_nd_cpu(pos_targs[si]["cls"], tf.stack([b_idx, g_idx], axis=1))  # [M, C]
                    cls_loss = tf.reduce_sum(bce_with_logits_loss(pred_cls_sel, tgt_cls)) / num_pos
                    total_cls += cls_loss

                    # Regression and DFL on positives
                    pred_reg_sel = self._gather_cpu(reg_map, b_idx)
                    pred_reg_sel = self._gather_cpu(pred_reg_sel, lin_idx, batch_dims=1)  # [M, 4*(R)]
                    bins = self.cfg.reg_max + 1
                    pred_reg_sel = tf.reshape(pred_reg_sel, [-1, 4, bins])
                    tgt_ltrb = tf.gather_nd(pos_targs[si]["ltrb"], tf.stack([b_idx, g_idx], axis=1))  # [M, 4]
                    dfl = dfl_loss(pred_reg_sel, tgt_ltrb, self.cfg.reg_max)
                    total_dfl += dfl

                    # decode boxes for IoU loss
                    # compute point coords for selected indices
                    pts_sel = tf.gather(pts, lin_idx)  # [M, 2] (no gradient to model outputs)
                    # distances in strides
                    dist = tgt_ltrb * stride
                    x1 = pts_sel[:, 0] - dist[:, 0]
                    y1 = pts_sel[:, 1] - dist[:, 1]
                    x2 = pts_sel[:, 0] + dist[:, 2]
                    y2 = pts_sel[:, 1] + dist[:, 3]
                    pred_dist = integral_distribution(tf.expand_dims(pred_reg_sel, 0), self.cfg.reg_max)[0]
                    pred_dist_pix = pred_dist * stride
                    px1 = pts_sel[:, 0] - pred_dist_pix[:, 0]
                    py1 = pts_sel[:, 1] - pred_dist_pix[:, 1]
                    px2 = pts_sel[:, 0] + pred_dist_pix[:, 2]
                    py2 = pts_sel[:, 1] + pred_dist_pix[:, 3]
                    box_loss = tf.reduce_mean(
                        bbox_ciou(tf.stack([px1, py1, px2, py2], axis=-1), tf.stack([x1, y1, x2, y2], axis=-1))
                    )
                    total_box += box_loss

                    # Objectness on positives
                    pred_obj_pos = self._gather_cpu(obj_map, b_idx)
                    pred_obj_pos = self._gather_cpu(pred_obj_pos, lin_idx, batch_dims=1)  # [M,1]
                    obj_pos_tgt = tf.ones_like(pred_obj_pos)
                    obj_pos_loss = tf.reduce_mean(bce_with_logits_loss(pred_obj_pos, obj_pos_tgt))
                    total_obj += obj_pos_loss

                # Center-radius assignment for extra cls/obj positives with IoU weighting (task-aligned)
                r = tf.constant(2.5, dtype=tf.float32)
                radius_pix = r * stride
                # Prepare targets fields
                t_cls_full = tf.cast(targets[..., 0], tf.int32)  # [B,maxb]
                tx1 = targets[..., 1:2]
                ty1 = targets[..., 2:3]
                tx2 = targets[..., 3:4]
                ty2 = targets[..., 4:5]
                tvalid = targets[..., 5:6] > 0.5
                # centers and distances
                tcx = (tx1 + tx2) / 2.0
                tcy = (ty1 + ty2) / 2.0
                centers = tf.concat([tcx, tcy], axis=-1)  # [B,maxb,2]
                pts_b = tf.tile(pts[None, ...], [B, 1, 1])  # [B,N,2]
                diff = tf.expand_dims(pts_b, axis=1) - tf.expand_dims(centers, axis=2)  # [B,maxb,N,2]
                d_all = tf.norm(diff, axis=-1)  # [B,maxb,N]
                # inside mask
                pts_x = tf.expand_dims(pts_b[..., 0], axis=1)  # [B,1,N]
                pts_y = tf.expand_dims(pts_b[..., 1], axis=1)
                x1e = tx1  # [B,maxb,1]
                y1e = ty1
                x2e = tx2
                y2e = ty2
                inside = tf.logical_and(tf.logical_and(pts_x >= x1e, pts_x <= x2e),
                                         tf.logical_and(pts_y >= y1e, pts_y <= y2e))  # [B,maxb,N]
                allow = tf.logical_and(inside, tvalid)
                allow = tf.logical_and(allow, d_all <= radius_pix)
                big = tf.constant(1e9, dtype=d_all.dtype)
                d_masked = tf.where(allow, d_all, tf.fill(tf.shape(d_all), big))
                any_ok = tf.reduce_any(allow, axis=1)  # [B,N]
                gt_choice = tf.argmin(d_masked, axis=1, output_type=tf.int32)  # [B,N]
                pos_pairs = tf.where(any_ok)  # [M,2] (b, n)
                def _cr_additions():
                    b_pos = tf.cast(pos_pairs[:, 0], tf.int32)
                    n_pos = tf.cast(pos_pairs[:, 1], tf.int32)
                    g_pos = tf.gather_nd(gt_choice, pos_pairs)  # [M]
                    gi = tf.stack([b_pos, g_pos], axis=1)
                    # class one-hot
                    cls_ids_pos = tf.gather_nd(t_cls_full, gi)
                    cls_oh_pos = tf.one_hot(cls_ids_pos, depth=C)
                    # Predicted boxes for selected points (DFL decode)
                    bins = self.cfg.reg_max + 1
                    n_pos = self._assert_indices_in_range(n_pos, HW, name="n_pos")
                    pred_reg_pts = self._gather_cpu(reg_map, b_pos)
                    pred_reg_pts = self._gather_cpu(pred_reg_pts, n_pos, batch_dims=1)  # [M, 4*(R)]
                    pred_reg_pts = tf.reshape(pred_reg_pts, [-1, 4, bins])
                    dist_bins = integral_distribution(pred_reg_pts, self.cfg.reg_max) * stride  # [M,4]
                    pxy = tf.gather(pts, n_pos)  # [M,2] (no gradient to model outputs)
                    px1 = pxy[:, 0] - dist_bins[:, 0]
                    py1 = pxy[:, 1] - dist_bins[:, 1]
                    px2 = pxy[:, 0] + dist_bins[:, 2]
                    py2 = pxy[:, 1] + dist_bins[:, 3]
                    pbox = tf.stack([px1, py1, px2, py2], axis=-1)  # [M,4]
                    # GT boxes for selected pairs
                    gx1 = tf.gather_nd(tf.squeeze(tx1, -1), gi)
                    gy1 = tf.gather_nd(tf.squeeze(ty1, -1), gi)
                    gx2 = tf.gather_nd(tf.squeeze(tx2, -1), gi)
                    gy2 = tf.gather_nd(tf.squeeze(ty2, -1), gi)
                    gbox = tf.stack([gx1, gy1, gx2, gy2], axis=-1)
                    # IoU weights
                    inter_x1 = tf.maximum(pbox[:, 0], gbox[:, 0])
                    inter_y1 = tf.maximum(pbox[:, 1], gbox[:, 1])
                    inter_x2 = tf.minimum(pbox[:, 2], gbox[:, 2])
                    inter_y2 = tf.minimum(pbox[:, 3], gbox[:, 3])
                    iw = tf.maximum(0.0, inter_x2 - inter_x1)
                    ih = tf.maximum(0.0, inter_y2 - inter_y1)
                    inter = iw * ih
                    area_p = tf.maximum(0.0, (pbox[:, 2] - pbox[:, 0])) * tf.maximum(0.0, (pbox[:, 3] - pbox[:, 1]))
                    area_g = tf.maximum(0.0, (gbox[:, 2] - gbox[:, 0])) * tf.maximum(0.0, (gbox[:, 3] - gbox[:, 1]))
                    union = area_p + area_g - inter + 1e-9
                    iou_w = inter / union  # [M]
                    # Keep top-K by IoU; prefer GPU placement for speed
                    M = tf.shape(iou_w)[0]
                    Kp = tf.minimum(M, tf.constant(self.cfg.cr_topk_limit, dtype=tf.int32))
                    def _do_topk():
                        res = tf.math.top_k(iou_w, k=Kp)
                        return res.values, res.indices
                    def _empty():
                        return tf.zeros([0], dtype=iou_w.dtype), tf.zeros([0], dtype=tf.int32)
                    top_vals, top_idx = tf.cond(M > 0, _do_topk, _empty)
                    i_sel = top_idx
                    w = top_vals
                    pred_cls_pts = self._gather_cpu(self._gather_cpu(cls_map, b_pos), i_sel, batch_dims=0)
                    n_pos_sel = tf.gather(n_pos, i_sel)
                    n_pos_sel = self._assert_indices_in_range(n_pos_sel, HW, name="n_pos_sel")
                    pred_cls_pts = self._gather_cpu(pred_cls_pts, n_pos_sel, batch_dims=1)
                    cls_oh_sel = tf.gather(cls_oh_pos, i_sel)
                    pred_obj_pts = self._gather_cpu(self._gather_cpu(obj_map, b_pos), i_sel, batch_dims=0)
                    pred_obj_pts = self._gather_cpu(pred_obj_pts, n_pos_sel, batch_dims=1)
                    cls_loss_pts = bce_with_logits_loss(pred_cls_pts, cls_oh_sel)
                    cls_loss_pts = tf.reduce_sum(cls_loss_pts * tf.where(tf.size(w) > 0, w[:, None], tf.zeros_like(cls_loss_pts))) / (tf.reduce_sum(w) + 1e-9)
                    obj_pos_loss2 = tf.reduce_sum(bce_with_logits_loss(pred_obj_pts, tf.ones_like(pred_obj_pts)) * tf.where(tf.size(w) > 0, w[:, None], tf.zeros_like(pred_obj_pts))) / (tf.reduce_sum(w) + 1e-9)
                    return cls_loss_pts, obj_pos_loss2
                def _no_cr():
                    z = tf.constant(0.0, tf.float32)
                    return z, z
                add_cls, add_obj = tf.cond(tf.shape(pos_pairs)[0] > 0, _cr_additions, _no_cr)
                total_cls += add_cls
                total_obj += add_obj

                # Background negative classification sampling to teach background separation
                # Build [B, HW] mask of positive points.
                # Build mask of positives; stop gradient since it's used for sampling only
                pos_points = tf.zeros([B, HW], dtype=tf.bool)
                m = tf.shape(b_idx)[0]
                def _do_scatter():
                    idxs = tf.stack([b_idx, lin_idx], axis=1)
                    updates = tf.ones([m], dtype=tf.bool)
                    if getattr(self.cfg, 'prefer_gpu_ops', True):
                        return tf.scatter_nd(idxs, updates, [B, HW])
                    with tf.device('/CPU:0'):
                        return tf.scatter_nd(idxs, updates, [B, HW])
                pos_points = tf.cond(m > 0, _do_scatter, lambda: pos_points)
                pos_points = tf.stop_gradient(pos_points)
                neg_mask_points = tf.logical_not(pos_points)
                # Sample up to K negatives per image by top-k random scores over negatives
                K = tf.minimum(HW, tf.constant(512, dtype=tf.int32))
                rnd = tf.random.uniform([B, HW], dtype=tf.float32)
                scores = tf.where(neg_mask_points, rnd, tf.fill([B, HW], -1.0))
                _, neg_idx = tf.math.top_k(scores, k=K)
                if getattr(self.cfg, 'debug_asserts', False):
                    _ = self._assert_indices_in_range(neg_idx, HW, name="neg_idx")
                pred_neg_cls = self._gather_cpu(cls_map, neg_idx, batch_dims=1)  # [B, K, C]
                zero_tgt = tf.zeros_like(pred_neg_cls)
                neg_ce = bce_with_logits_loss(pred_neg_cls, zero_tgt)
                neg_cls_loss = 0.25 * tf.reduce_mean(neg_ce)
                total_cls += neg_cls_loss

                # Objectness negatives
                pred_neg_obj = self._gather_cpu(obj_map, neg_idx, batch_dims=1)  # [B,K,1]
                zero_obj = tf.zeros_like(pred_neg_obj)
                neg_obj_loss = 0.25 * tf.reduce_mean(bce_with_logits_loss(pred_neg_obj, zero_obj))
                total_obj += neg_obj_loss

                # Skip expensive full decode during training; evaluation runs its own decode

            # combine losses
            loss = (
                self.cfg.cls_loss_gain * total_cls
                + self.cfg.box_loss_gain * total_box
                + self.cfg.dfl_loss_gain * total_dfl
                + 1.0 * total_obj
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return {
            "loss": tf.cast(loss, tf.float32),
            "cls": tf.cast(total_cls, tf.float32),
            "box": tf.cast(total_box, tf.float32),
            "dfl": tf.cast(total_dfl, tf.float32),
            "pos": tf.cast(total_pos, tf.float32),
            "obj": tf.cast(total_obj, tf.float32),
        }


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
