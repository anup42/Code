from typing import List

import numpy as np
import tensorflow as tf
from .losses import integral_distribution
from .inference import YoloInferencer


def _xywhn_to_xyxy_np(xywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def _bbox_iou_np(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-9) -> np.ndarray:
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


def decode_maps_to_dets_np(pred, conf_thres=0.25, iou_thres=0.5, max_det=300, imgsz: int | None = None) -> List[np.ndarray]:
    """Decode predictions to detections.

    Supports two formats:
      1) dict with keys 'cls', 'reg', 'grids', 'strides' (DFL-style)
      2) list of 3 maps [B,H,W,5+C] (legacy xywh+obj)
    Returns: list length B with arrays [K,6] = [x1,y1,x2,y2,score,cls] normalized 0..1.
    """
    # DFL-style
    if isinstance(pred, dict) and all(k in pred for k in ("cls", "reg", "grids", "strides")):
        cls_list = [p.numpy() if isinstance(p, tf.Tensor) else p for p in pred["cls"]]
        reg_list = [p.numpy() if isinstance(p, tf.Tensor) else p for p in pred["reg"]]
        grids = [g.numpy() if isinstance(g, tf.Tensor) else g for g in pred["grids"]]
        strides = pred["strides"]
        B = cls_list[0].shape[0]
        C = cls_list[0].shape[-1]
        outs = [[] for _ in range(B)]
        # Concatenate across scales
        cls_all = np.concatenate(cls_list, axis=1)  # [B,N,C]
        reg_all = np.concatenate(reg_list, axis=1)  # [B,N,4*bins]
        grid_all = np.concatenate(grids, axis=0)    # [N,2]
        # Build stride per point aligned with concatenation
        stride_vec = np.concatenate([np.full((g.shape[0],), float(strides[i]), dtype=np.float32) for i, g in enumerate(grids)], axis=0)  # [N]
        # Decode distances via integral and scale by stride to pixel units
        bins = reg_all.shape[-1] // 4
        reg_reshaped = reg_all.reshape([B, -1, 4, bins])
        dist = integral_distribution(tf.convert_to_tensor(reg_reshaped), bins - 1).numpy()  # [B,N,4]
        dist_pix = dist * stride_vec[None, :, None]  # [B,N,4]
        # Convert to xyxy in pixels then normalize by image size
        if imgsz is not None:
            img_w = float(imgsz)
            img_h = float(imgsz)
        else:
            # Fallback: infer from grid range
            min_stride = float(min(strides)) if hasattr(strides, '__iter__') else float(strides)
            img_w = float((grid_all[:, 0].max() - grid_all[:, 0].min()) + min_stride)
            img_h = float((grid_all[:, 1].max() - grid_all[:, 1].min()) + min_stride)
        for b in range(B):
            pts = grid_all  # [N,2]
            d = dist_pix[b]
            x1 = pts[:, 0] - d[:, 0]
            y1 = pts[:, 1] - d[:, 1]
            x2 = pts[:, 0] + d[:, 2]
            y2 = pts[:, 1] + d[:, 3]
            boxes = np.stack([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h], axis=-1)
            # class scores
            scores_c = 1.0 / (1.0 + np.exp(-cls_all[b]))  # [N,C]
            if scores_c.shape[0] == 0:
                outs[b] = np.zeros((0, 6), dtype=np.float32)
                continue
            # light prefilter: keep many candidates to avoid dropping TPs early
            topk = min(max_det * 50, scores_c.shape[0])
            topk = max(1, topk)
            max_cls = scores_c.max(axis=1)
            top_idx = np.argpartition(-max_cls, kth=topk - 1)[:topk]
            boxes_f = boxes[top_idx]
            scores_c_f = scores_c[top_idx]
            dets_b = []
            for c in range(C):
                sc = scores_c_f[:, c]
                mask = sc >= conf_thres
                if not np.any(mask):
                    continue
                b_c = boxes_f[mask]
                s_c = sc[mask]
                keep = _nms_np(b_c, s_c, iou_thres, max_det)
                if keep.size > 0:
                    kept = np.concatenate([b_c[keep], s_c[keep, None], np.full((keep.size, 1), c, dtype=np.float32)], axis=1)
                    dets_b.append(kept)
            if dets_b:
                merged = np.vstack(dets_b)
                if merged.shape[0] > max_det:
                    idx = np.argpartition(-merged[:, 4], kth=max_det - 1)[:max_det]
                    merged = merged[idx]
                outs[b] = merged
            else:
                outs[b] = np.zeros((0, 6), dtype=np.float32)
        return outs

    # Legacy xywh+obj list of maps
    pred_maps = pred
    B = pred_maps[0].shape[0]
    C = pred_maps[0].shape[-1] - 5
    outs = [[] for _ in range(B)]
    all_boxes = []
    all_obj = []
    all_cls = []
    for p in pred_maps:
        H, W = p.shape[1], p.shape[2]
        p = p.reshape(B, H * W, 5 + C)
        box_logits = p[..., 0:4]
        obj_logits = p[..., 4]
        cls_logits = p[..., 5:]
        box_sig = 1.0 / (1.0 + np.exp(-box_logits))
        obj = 1.0 / (1.0 + np.exp(-obj_logits))
        cls = 1.0 / (1.0 + np.exp(-cls_logits))
        all_boxes.append(box_sig)
        all_obj.append(obj)
        all_cls.append(cls)
    boxes = np.concatenate(all_boxes, axis=1)
    obj = np.concatenate(all_obj, axis=1)
    cls = np.concatenate(all_cls, axis=1)

    for b in range(B):
        bx = boxes[b]
        ob = obj[b]
        cl = cls[b]
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
        outs[b] = np.vstack(dets_b) if dets_b else np.zeros((0, 6), dtype=np.float32)
    return outs


def _compute_ap_curve(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0 or precision.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _gather_image_stats(dets: np.ndarray, gts: np.ndarray, num_classes: int, iou_thresholds: np.ndarray):
    num_iou = iou_thresholds.size
    if dets.size == 0:
        tp = np.zeros((0, num_iou), dtype=bool)
        conf = np.zeros((0,), dtype=np.float32)
        pred_cls = np.zeros((0,), dtype=np.int32)
    else:
        boxes = dets[:, 0:4].astype(np.float32)
        conf = dets[:, 4].astype(np.float32)
        pred_cls = dets[:, 5].astype(np.int32)
        if num_classes > 0:
            pred_cls = np.clip(pred_cls, 0, num_classes - 1)
        tp = np.zeros((boxes.shape[0], num_iou), dtype=bool)
        order = np.argsort(-conf)
        if gts.size > 0:
            gt_cls = gts[:, 0].astype(np.int32)
            gt_boxes = _xywhn_to_xyxy_np(gts[:, 1:5])
            ious = _bbox_iou_np(boxes, gt_boxes)
            matched = np.zeros((num_iou, gt_boxes.shape[0]), dtype=bool)
            for det_idx in order:
                cls = pred_cls[det_idx]
                gt_mask = np.where(gt_cls == cls)[0]
                if gt_mask.size == 0:
                    continue
                ious_cls = ious[det_idx, gt_mask]
                best_rel = int(np.argmax(ious_cls))
                best_gt = gt_mask[best_rel]
                best_iou = ious_cls[best_rel]
                for thr_idx, thr in enumerate(iou_thresholds):
                    if best_iou >= thr and not matched[thr_idx, best_gt]:
                        matched[thr_idx, best_gt] = True
                        tp[det_idx, thr_idx] = True
        tp = tp[order]
        conf = conf[order]
        pred_cls = pred_cls[order]
    target_cls = gts[:, 0].astype(np.int32) if gts.size > 0 else np.zeros((0,), dtype=np.int32)
    return tp, conf, pred_cls, target_cls


def _accumulate_stats(dets_by_img: List[np.ndarray], gts_by_img: List[np.ndarray], num_classes: int, iou_thresholds: np.ndarray):
    tp_list = []
    conf_list = []
    pred_cls_list = []
    target_cls_list = []
    for g, d in zip(gts_by_img, dets_by_img):
        tp, conf, pred_cls, target_cls = _gather_image_stats(d, g, num_classes, iou_thresholds)
        tp_list.append(tp)
        conf_list.append(conf)
        pred_cls_list.append(pred_cls)
        target_cls_list.append(target_cls)

    tp_all = np.concatenate(tp_list, axis=0) if tp_list else np.zeros((0, iou_thresholds.size), dtype=bool)
    conf_all = np.concatenate(conf_list, axis=0) if conf_list else np.zeros((0,), dtype=np.float32)
    pred_cls_all = np.concatenate(pred_cls_list, axis=0) if pred_cls_list else np.zeros((0,), dtype=np.int32)
    target_cls_all = np.concatenate(target_cls_list, axis=0) if target_cls_list else np.zeros((0,), dtype=np.int32)
    return tp_all, conf_all, pred_cls_all, target_cls_all


def _compute_precision_recall_ap(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray,
                                 target_cls: np.ndarray, num_classes: int, iou_thresholds: np.ndarray):
    num_iou = iou_thresholds.size
    ap = np.zeros((num_classes, num_iou), dtype=np.float32)
    class_counts = np.bincount(target_cls, minlength=num_classes).astype(np.int32) if num_classes > 0 else np.zeros((0,), dtype=np.int32)

    if tp.size:
        tp50 = tp[:, 0].astype(np.float32)
        total_tp = float(tp50.sum())
        total_fp = float(tp.shape[0] - tp50.sum())
    else:
        tp50 = np.zeros((0,), dtype=np.float32)
        total_tp = 0.0
        total_fp = 0.0
    total_det = total_tp + total_fp
    precision_global = float(total_tp / (total_det + 1e-9)) if total_det > 0 else 0.0
    recall_global = float(total_tp / (target_cls.size + 1e-9)) if target_cls.size > 0 else 0.0

    for c in range(num_classes):
        cls_mask = pred_cls == c
        n_gt = int(class_counts[c]) if c < class_counts.size else 0
        if cls_mask.sum() == 0:
            continue
        tp_c = tp[cls_mask].astype(np.float32)
        conf_c = conf[cls_mask].astype(np.float32)
        order = np.argsort(-conf_c)
        tp_c = tp_c[order]
        fp_c = (1.0 - tp_c).cumsum(axis=0)
        tp_cum = tp_c.cumsum(axis=0)
        if n_gt > 0:
            recall_curve = tp_cum / (n_gt + 1e-9)
            precision_curve = tp_cum / (tp_cum + fp_c + 1e-9)
            for j in range(num_iou):
                ap[c, j] = _compute_ap_curve(recall_curve[:, j], precision_curve[:, j])
        else:
            # No GT for this class -> AP stays zero
            continue

    return precision_global, recall_global, ap, class_counts


def compute_pr_map50(dets_by_img: List[np.ndarray], gts_by_img: List[np.ndarray], num_classes: int, iou_thres=0.5):
    iou_thresholds = np.array([iou_thres], dtype=np.float32)
    tp, conf, pred_cls, target_cls = _accumulate_stats(dets_by_img, gts_by_img, num_classes, iou_thresholds)
    precision, recall, ap, class_counts = _compute_precision_recall_ap(tp, conf, pred_cls, target_cls, num_classes, iou_thresholds)
    valid = class_counts > 0
    map50 = float(ap[valid, 0].mean()) if valid.any() else 0.0
    return precision, recall, map50


def compute_map50_95(dets_by_img: List[np.ndarray], gts_by_img: List[np.ndarray], num_classes: int):
    iou_thresholds = np.linspace(0.5, 0.95, 10, dtype=np.float32)
    tp, conf, pred_cls, target_cls = _accumulate_stats(dets_by_img, gts_by_img, num_classes, iou_thresholds)
    _, _, ap, class_counts = _compute_precision_recall_ap(tp, conf, pred_cls, target_cls, num_classes, iou_thresholds)
    valid = class_counts > 0
    return float(ap[valid].mean()) if valid.any() else 0.0


def evaluate_dataset_pr_maps(model: tf.keras.Model, val_ds, num_classes: int,
                             conf_thres=0.25, iou_thres=0.5, max_det=300, imgsz: int | None = None):
    """Compute precision, recall, mAP@0.5, and mAP@0.5:0.95."""
    dets_all: List[np.ndarray] = []
    gts_all: List[np.ndarray] = []
    for batch in val_ds:
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            (images, labels), _targets = batch
            preds = model(images, training=False)
            dets = decode_maps_to_dets_np(preds, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, imgsz=imgsz)
            labs = labels.numpy()
            for i in range(labs.shape[0]):
                li = labs[i]
                mask = (li[:, 3] > 0) & (li[:, 4] > 0)
                gts_all.append(li[mask])
                dets_all.append(dets[i])
        else:
            images, targets = batch
            preds = model(images, training=False)
            dets = decode_maps_to_dets_np(preds, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, imgsz=imgsz)
            imgs = images.numpy()
            H, W = imgs.shape[1], imgs.shape[2]
            tt = targets.numpy()
            for i in range(tt.shape[0]):
                t = tt[i]
                mask = t[:, 5] > 0.5
                t = t[mask]
                if t.size == 0:
                    gts_all.append(np.zeros((0, 5), dtype=np.float32))
                    dets_all.append(dets[i])
                    continue
                cls = t[:, 0:1]
                x1, y1, x2, y2 = t[:, 1:2], t[:, 2:3], t[:, 3:4], t[:, 4:5]
                cx = (x1 + x2) / 2.0 / float(W)
                cy = (y1 + y2) / 2.0 / float(H)
                w = (x2 - x1) / float(W)
                h = (y2 - y1) / float(H)
                gts_all.append(np.concatenate([cls, cx, cy, w, h], axis=1))
                dets_all.append(dets[i])

    iou_thresholds = np.linspace(0.5, 0.95, 10, dtype=np.float32)
    tp, conf, pred_cls, target_cls = _accumulate_stats(dets_all, gts_all, num_classes, iou_thresholds)
    precision, recall, ap, class_counts = _compute_precision_recall_ap(tp, conf, pred_cls, target_cls, num_classes, iou_thresholds)
    valid = class_counts > 0
    map50 = float(ap[valid, 0].mean()) if valid.any() else 0.0
    map5095 = float(ap[valid].mean()) if valid.any() else 0.0
    return precision, recall, map50, map5095


def evaluate_dataset_pr_maps_fast(
    model: tf.keras.Model,
    val_ds,
    num_classes: int,
    conf_thres=0.25,
    iou_thres=0.5,
    max_det=300,
    imgsz: int | None = None,
    progbar=None,
    total_steps: int | None = None,
    trainer=None,
    return_loss: bool = False,
):
    """Faster evaluation using YoloInferencer + combined_nms and optional progress bar.

    If progbar is provided (e.g., tf.keras.utils.Progbar), updates it each batch up to total_steps.
    """
    infer = YoloInferencer(model, score_thresh=conf_thres, iou_thresh=iou_thres)

    dets_all: List[np.ndarray] = []
    gts_all: List[np.ndarray] = []
    val_loss_totals = None
    val_batches = 0

    if return_loss and trainer is not None:
        val_loss_totals = {key: 0.0 for key in ("loss", "cls", "box", "dfl", "obj", "pos")}

    step = 0
    for batch in val_ds:
        step += 1
        if total_steps is not None and step > total_steps:
            break

        if trainer is not None:
            images, targets = trainer.extract_images_targets(batch)
            outputs = trainer.model(images, training=False)
            boxes_t, scores_t, classes_t, valid_t = infer.predict_from_outputs(
                outputs,
                tf.shape(images)[1],
                tf.shape(images)[2],
            )

            if return_loss and val_loss_totals is not None:
                val_metrics = trainer.compute_loss_metrics(outputs, targets)
                val_batches += 1
                for key in val_loss_totals:
                    val_loss_totals[key] += float(val_metrics[key].numpy())

            dyn_shape = tf.shape(images)
            B = int(images.shape[0]) if images.shape[0] is not None else int(dyn_shape[0].numpy())
            H = int(images.shape[1]) if images.shape[1] is not None else int(dyn_shape[1].numpy())
            W = int(images.shape[2]) if images.shape[2] is not None else int(dyn_shape[2].numpy())
            for i in range(B):
                n = int(valid_t[i].numpy())
                bi = boxes_t[i][:n].numpy()
                si = scores_t[i][:n].numpy()
                ci = classes_t[i][:n].numpy().astype(np.float32)
                if n > 0:
                    dets_all.append(np.concatenate([bi, si[:, None], ci[:, None]], axis=1))
                else:
                    dets_all.append(np.zeros((0, 6), dtype=np.float32))
            tt = targets.numpy()
            for i in range(tt.shape[0]):
                t = tt[i]
                mask = t[:, 5] > 0.5
                t = t[mask]
                if t.size == 0:
                    gts_all.append(np.zeros((0, 5), dtype=np.float32))
                else:
                    cls = t[:, 0:1]
                    x1, y1, x2, y2 = t[:, 1:2], t[:, 2:3], t[:, 3:4], t[:, 4:5]
                    cx = (x1 + x2) / 2.0 / float(W)
                    cy = (y1 + y2) / 2.0 / float(H)
                    w = (x2 - x1) / float(W)
                    h = (y2 - y1) / float(H)
                    gts_all.append(np.concatenate([cls, cx, cy, w, h], axis=1))
        else:
            if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (tuple, list)):
                (images, labels), _targets = batch
                boxes_t, scores_t, classes_t, valid_t = infer.predict(images)
                dyn_shape = tf.shape(images)
                B = int(images.shape[0]) if images.shape[0] is not None else int(dyn_shape[0].numpy())
                for i in range(B):
                    n = int(valid_t[i].numpy())
                    bi = boxes_t[i][:n].numpy()
                    si = scores_t[i][:n].numpy()
                    ci = classes_t[i][:n].numpy().astype(np.float32)
                    if n > 0:
                        dets_all.append(np.concatenate([bi, si[:, None], ci[:, None]], axis=1))
                    else:
                        dets_all.append(np.zeros((0, 6), dtype=np.float32))
                labs = labels.numpy()
                for i in range(labs.shape[0]):
                    li = labs[i]
                    mask = (li[:, 3] > 0) & (li[:, 4] > 0)
                    gts_all.append(li[mask])
            else:
                images, targets = batch
                boxes_t, scores_t, classes_t, valid_t = infer.predict(images)
                dyn_shape = tf.shape(images)
                B = int(images.shape[0]) if images.shape[0] is not None else int(dyn_shape[0].numpy())
                H = int(images.shape[1]) if images.shape[1] is not None else int(dyn_shape[1].numpy())
                W = int(images.shape[2]) if images.shape[2] is not None else int(dyn_shape[2].numpy())
                for i in range(B):
                    n = int(valid_t[i].numpy())
                    bi = boxes_t[i][:n].numpy()
                    si = scores_t[i][:n].numpy()
                    ci = classes_t[i][:n].numpy().astype(np.float32)
                    if n > 0:
                        dets_all.append(np.concatenate([bi, si[:, None], ci[:, None]], axis=1))
                    else:
                        dets_all.append(np.zeros((0, 6), dtype=np.float32))
                tt = targets.numpy()
                for i in range(tt.shape[0]):
                    t = tt[i]
                    mask = t[:, 5] > 0.5
                    t = t[mask]
                    if t.size == 0:
                        gts_all.append(np.zeros((0, 5), dtype=np.float32))
                    else:
                        cls = t[:, 0:1]
                        x1, y1, x2, y2 = t[:, 1:2], t[:, 2:3], t[:, 3:4], t[:, 4:5]
                        cx = (x1 + x2) / 2.0 / float(W)
                        cy = (y1 + y2) / 2.0 / float(H)
                        w = (x2 - x1) / float(W)
                        h = (y2 - y1) / float(H)
                        gts_all.append(np.concatenate([cls, cx, cy, w, h], axis=1))

        if progbar is not None and total_steps is not None:
            progbar.update(min(step, total_steps))

    iou_thresholds = np.linspace(0.5, 0.95, 10, dtype=np.float32)
    tp, conf, pred_cls, target_cls = _accumulate_stats(dets_all, gts_all, num_classes, iou_thresholds)
    precision, recall, ap, class_counts = _compute_precision_recall_ap(tp, conf, pred_cls, target_cls, num_classes, iou_thresholds)
    valid = class_counts > 0
    map50 = float(ap[valid, 0].mean()) if valid.any() else 0.0
    map5095 = float(ap[valid].mean()) if valid.any() else 0.0
    if return_loss and trainer is not None and val_loss_totals is not None:
        avg_losses = {k: v / max(1, val_batches) for k, v in val_loss_totals.items()}
        return precision, recall, map50, map5095, avg_losses
    return precision, recall, map50, map5095


def evaluate_dataset_map50(model: tf.keras.Model, val_ds, num_classes: int,
                           conf_thres=0.25, iou_thres=0.5, max_det=300, imgsz: int | None = None):
    dets_all: List[np.ndarray] = []
    gts_all: List[np.ndarray] = []
    for batch in val_ds:
        # Accept either ((images, labels), targets) or (images, targets)
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            (images, labels), _targets = batch
            preds = model(images, training=False)
            dets = decode_maps_to_dets_np(preds, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, imgsz=imgsz)
            labs = labels.numpy()
            for i in range(labs.shape[0]):
                li = labs[i]
                mask = (li[:, 3] > 0) & (li[:, 4] > 0)
                gts_all.append(li[mask])
                dets_all.append(dets[i])
        else:
            images, targets = batch
            preds = model(images, training=False)
            dets = decode_maps_to_dets_np(preds, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, imgsz=imgsz)
            imgs = images.numpy()
            H, W = imgs.shape[1], imgs.shape[2]
            tt = targets.numpy()  # [B, max_labels, 6]
            for i in range(tt.shape[0]):
                t = tt[i]
                mask = t[:, 5] > 0.5
                t = t[mask]
                if t.size == 0:
                    gts_all.append(np.zeros((0, 5), dtype=np.float32))
                    dets_all.append(dets[i])
                    continue
                cls = t[:, 0:1]
                x1, y1, x2, y2 = t[:, 1:2], t[:, 2:3], t[:, 3:4], t[:, 4:5]
                cx = (x1 + x2) / 2.0 / float(W)
                cy = (y1 + y2) / 2.0 / float(H)
                w = (x2 - x1) / float(W)
                h = (y2 - y1) / float(H)
                gts_all.append(np.concatenate([cls, cx, cy, w, h], axis=1))
                dets_all.append(dets[i])
    return evaluate_pr_map50_from_lists(dets_all, gts_all, num_classes, iou_thres=iou_thres)


def evaluate_pr_map50_from_lists(dets_all, gts_all, num_classes, iou_thres=0.5):
    return compute_pr_map50(dets_all, gts_all, num_classes, iou_thres=iou_thres)
