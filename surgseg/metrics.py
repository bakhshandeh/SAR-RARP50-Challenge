import numpy as np
import torch
import cv2
from .config import IGNORE_LABEL

def compute_confusion_matrix(pred, target, num_classes, ignore_index=IGNORE_LABEL):
    mask = target != ignore_index
    pred = pred[mask].view(-1)
    target = target[mask].view(-1)
    k = (target >= 0) & (target < num_classes)
    inds = num_classes * target[k] + pred[k]
    cm = torch.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes).cpu().numpy()
    return cm

def miou_from_cm(cm):
    ious = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
    return float(np.mean(ious)), ious

def _boundary_mask(label, cls, thickness=1):
    m = (label == cls).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(m, kernel, iterations=1)
    ero = cv2.erode(m, kernel, iterations=1)
    boundary = cv2.subtract(dil, ero)
    if thickness > 1:
        boundary = cv2.dilate(boundary, kernel, iterations=thickness-1)
    return boundary

def mnsd(pred, target, num_classes, tau=3, ignore_index=IGNORE_LABEL):
    valid = (target != ignore_index).astype(np.uint8)
    per_class = []
    for k in range(num_classes):
        b_pred = _boundary_mask(pred, k) * valid
        b_gt   = _boundary_mask(target, k) * valid
        if b_pred.sum() == 0 and b_gt.sum() == 0:
            per_class.append(1.0); continue
        if b_pred.sum() == 0 or b_gt.sum() == 0:
            per_class.append(0.0); continue
        dt_gt = cv2.distanceTransform(1 - b_gt, cv2.DIST_L2, 3)
        dt_pred = cv2.distanceTransform(1 - b_pred, cv2.DIST_L2, 3)
        pred_dists = dt_gt[b_pred.astype(bool)]
        gt_dists   = dt_pred[b_gt.astype(bool)]
        delta_pred = (pred_dists <= tau).sum()
        delta_gt   = (gt_dists   <= tau).sum()
        denom = len(pred_dists) + len(gt_dists)
        nsd_k = (delta_pred + delta_gt) / denom if denom > 0 else 0.0
        per_class.append(nsd_k)
    return float(np.mean(per_class)), per_class
