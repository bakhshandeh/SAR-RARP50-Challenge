import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import IGNORE_LABEL

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets, ignore_index=IGNORE_LABEL):
        if targets.ndim == 4 and targets.size(1) == 1:
            targets = targets[:, 0, ...]
        elif targets.ndim != 3:
            raise ValueError(f"targets must be (B,H,W) or (B,1,H,W); got {tuple(targets.shape)}")
        num_classes = logits.shape[1]
        valid = targets != ignore_index
        probs = torch.softmax(logits, dim=1)
        idx = torch.clamp(targets, 0, num_classes - 1).unsqueeze(1)
        oh = torch.zeros_like(logits)
        oh.scatter_(1, idx, 1.0)
        oh = oh * valid.unsqueeze(1)
        probs = probs * valid.unsqueeze(1)
        dims = (0,2,3)
        numerator = 2.0 * (probs * oh).sum(dim=dims)
        denominator = (probs.pow(2).sum(dim=dims) + oh.pow(2).sum(dim=dims)).clamp_min(self.eps)
        dice = numerator / denominator
        return 1.0 - dice.mean()

class BoundaryDiceLoss(nn.Module):
    """Edge-focused dice using local max-min to approximate boundaries."""
    def __init__(self, radius=1, eps=1e-5):
        super().__init__()
        self.radius = radius
        self.eps = eps
    def _boundary_map(self, x):
        pad = self.radius
        maxp = F.max_pool2d(x, kernel_size=2*pad+1, stride=1, padding=pad)
        minp = -F.max_pool2d(-x, kernel_size=2*pad+1, stride=1, padding=pad)
        return (maxp - minp).clamp(0, 1)
    def forward(self, logits, targets, ignore_index=IGNORE_LABEL):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        with torch.no_grad():
            t = torch.clamp(targets, 0, num_classes-1)
            valid = (targets != ignore_index).float()
            oh = torch.zeros_like(logits).scatter_(1, t.unsqueeze(1), 1.0)
            oh = oh * valid.unsqueeze(1)
        pred_b = self._boundary_map(probs)
        gt_b   = self._boundary_map(oh)
        dims = (0,2,3)
        inter = (pred_b * gt_b).sum(dim=dims)
        denom = (pred_b.pow(2).sum(dim=dims) + gt_b.pow(2).sum(dim=dims)).clamp_min(self.eps)
        dice = 2*inter/denom
        return 1.0 - dice.mean()

# NOTE:
# For thin or elongated structures (e.g., surgical threads, catheters, vessels),
# adjusting and increasing `boundary_weight` can significantly improve results.
# This is because boundary-focused loss terms better capture fine edges and
# reduce errors where small structures might otherwise be missed or thickened.
class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None,
                 dice_weight=0.25, ce_weight=0.75,
                 boundary_weight=0.0, boundary_radius=1,
                 ignore_index=IGNORE_LABEL):
        super().__init__()
        self.dice = DiceLoss()
        self.boundary = BoundaryDiceLoss(radius=boundary_radius)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dw, self.cw, self.bw = dice_weight, ce_weight, boundary_weight
        self.ignore_index = ignore_index
    def forward(self, logits, targets):
        return (
            self.cw * self.ce(logits, targets) +
            self.dw * self.dice(logits, targets, self.ignore_index) +
            self.bw * self.boundary(logits, targets, self.ignore_index)
        )
