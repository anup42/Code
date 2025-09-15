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
            # light prefilter: keep many candidates to avoid dropping TPs early
            topk = min(max_det * 50, scores_c.shape[0])
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


def compute_pr_map50(dets_by_img: List[np.ndarray], gts_by_img: List[np.ndarray], num_classes: int, iou_thres=0.5):
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


def compute_map_at_iou(dets_by_img: List[np.ndarray], gts_by_img: List[np.ndarray], num_classes: int, iou_thres=0.5):
    """Compute mAP at a specific IoU threshold (COCO-style AP)."""
    ap_list = []
    for c in range(num_classes):
        scores_c = []
        labels_c = []
        pos = 0
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

            gt_idx = np.where(g_cls == c)[0]
            det_idx = np.where(d_cls == c)[0]
            pos += gt_idx.size
            if det_idx.size == 0:
                continue
            order = det_idx[np.argsort(-d_scores[det_idx])]
            matched = np.zeros(gt_idx.size, dtype=bool)
            for di in order:
                if gt_idx.size == 0:
                    scores_c.append(d_scores[di])
                    labels_c.append(0)
                    continue
                ious = _bbox_iou_np(d_xyxy[di:di+1], g_xyxy[gt_idx])[0]
                best = np.argmax(ious)
                if ious[best] >= iou_thres and not matched[best]:
                    matched[best] = True
                    scores_c.append(d_scores[di])
                    labels_c.append(1)
                else:
                    scores_c.append(d_scores[di])
                    labels_c.append(0)
        if len(scores_c) == 0 or pos == 0:
            ap_list.append(0.0)
            continue
        scores = np.array(scores_c, dtype=np.float32)
        labels = np.array(labels_c, dtype=np.int32)
        order = np.argsort(-scores)
        labels = labels[order]
        tp_cum = np.cumsum(labels)
        fp_cum = np.cumsum(1 - labels)
        recall = tp_cum / float(pos)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        ap_list.append(float(ap))
    return float(np.mean(ap_list)) if len(ap_list) else 0.0


def compute_map50_95(dets_by_img: List[np.ndarray], gts_by_img: List[np.ndarray], num_classes: int):
    ious = [0.5 + 0.05 * i for i in range(10)]
    aps = [compute_map_at_iou(dets_by_img, gts_by_img, num_classes, iou) for iou in ious]
    return float(np.mean(aps))


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

    p, r, map50 = compute_pr_map50(dets_all, gts_all, num_classes, iou_thres=0.5)
    map5095 = compute_map50_95(dets_all, gts_all, num_classes)
    return p, r, map50, map5095


def evaluate_dataset_pr_maps_fast(model: tf.keras.Model, val_ds, num_classes: int,
                                  conf_thres=0.25, iou_thres=0.5, max_det=300, imgsz: int | None = None,
                                  progbar=None, total_steps: int | None = None):
    """Faster evaluation using YoloInferencer + combined_nms and optional progress bar.

    If progbar is provided (e.g., tf.keras.utils.Progbar), updates it each batch up to total_steps.
    """
    assert imgsz is not None, "imgsz is required for normalization in fast evaluation"
    infer = YoloInferencer(model, score_thresh=conf_thres, iou_thresh=iou_thres)

    dets_all: List[np.ndarray] = []
    gts_all: List[np.ndarray] = []
    step = 0
    for batch in val_ds:
        step += 1
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            (images, labels), _targets = batch
            boxes, scores, classes, valid = infer.predict(images)
            B = images.shape[0]
            for i in range(B):
                n = int(valid[i].numpy())
                bi = boxes[i][:n].numpy() / float(imgsz)
                si = scores[i][:n].numpy()
                ci = classes[i][:n].numpy().astype(np.float32)
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
            boxes, scores, classes, valid = infer.predict(images)
            B = images.shape[0]
            H, W = images.shape[1], images.shape[2]
            for i in range(B):
                n = int(valid[i].numpy())
                bi = boxes[i][:n].numpy() / float(imgsz)
                si = scores[i][:n].numpy()
                ci = classes[i][:n].numpy().astype(np.float32)
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

    p, r, map50 = compute_pr_map50(dets_all, gts_all, num_classes, iou_thres=0.5)
    map5095 = compute_map50_95(dets_all, gts_all, num_classes)
    return p, r, map50, map5095


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
