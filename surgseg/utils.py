import os
import random
import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
import cv2

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    torch.backends.cudnn.benchmark = True
    return dev

def save_ckpt(path, model, optimizer, epoch, best):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best": best
    }, path)

def parse_size(s: str):
    w, h = [int(x) for x in s.lower().split("x")]
    return (w, h)

# ---------- Inference image utilities ----------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def letterbox_rgb(rgb: np.ndarray, target_size):
    """
    Letterbox an RGB image to (W,H) while keeping aspect ratio.
    Returns (rgb_proc, offsets=(off_y, off_x, new_h, new_w)).
    """
    h0, w0 = rgb.shape[:2]
    tw, th = target_size
    scale = min(tw / w0, th / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = resized
    return canvas, (oy, ox, nh, nw)

def load_image_as_tensor(path: Path, target_size=None):
    """
    Read an image path -> (tensor[BCHW, float, normalized], rgb_proc[H,W,3], meta).
    If target_size provided (W,H), letterbox the image; else keep original size.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = rgb.shape[:2]

    if target_size is not None:
        rgb_proc, offsets = letterbox_rgb(rgb, target_size)
        TH, TW = target_size[1], target_size[0]
    else:
        rgb_proc, offsets = rgb, None
        TH, TW = H0, W0

    x = rgb_proc.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))            # HWC -> CHW
    t = torch.from_numpy(x).unsqueeze(0)      # 1xCxHxW

    meta = {"orig": (H0, W0), "proc": (TH, TW), "offsets": offsets}
    return t, rgb_proc, meta

def resize_logits_to(logits: torch.Tensor, size_hw):
    """Bilinear resize logits to (H,W)."""
    return F.interpolate(logits, size=size_hw, mode="bilinear", align_corners=False)

# ---------- Visualization ----------

def color_palette(n):
    """
    Deterministic palette (first 9 colors fixed, rest random but seeded).
    Returns uint8 array [n,3] in RGB.
    """
    np.random.seed(42)
    base = np.array([
        [ 66,135,245],[245,66, 99],[ 66,245,149],
        [245,199, 66],[155,66,245],[ 66,245,233],
        [245, 66,188],[120,245, 66],[245,132, 66]
    ], dtype=np.uint8)
    if n <= len(base):
        return base[:n]
    extra = np.random.randint(0, 256, size=(n - len(base), 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0)

def overlay_mask(image_rgb_uint8: np.ndarray, mask_hw: np.ndarray, alpha=0.5, num_classes=9):
    """Overlay argmax mask onto an RGB image."""
    h, w = mask_hw.shape
    pal = color_palette(num_classes)
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for k in range(num_classes):
        color[mask_hw == k] = pal[k]
    return cv2.addWeighted(image_rgb_uint8, 1.0 - alpha, color, alpha, 0.0)

# ---------- Checkpoint IO ----------

def load_checkpoint_state(model: torch.nn.Module, ckpt_path: Path, strict=False, verbose=False):
    """
    Load a checkpoint that may store either the raw state_dict or a dict with 'state_dict'.
    Returns (missing_keys, unexpected_keys).
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if (missing or unexpected) and verbose:
        print(f"[utils] load_state: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:   print("  missing:", missing)
        if unexpected:print("  unexpected:", unexpected)
    return missing, unexpected

# ---------- Postprocess helpers ----------

def undo_letterbox_and_upscale(pred_mask: np.ndarray, frame_path: Path, meta: dict, undo_letterbox: bool):
    """
    Bring the predicted (letterboxed) mask back to original frame size.
    If undo_letterbox=True, crop to the content before upscaling.
    Returns (pred_mask_original_size, original_rgb)
    """
    H0, W0 = meta["orig"]
    if meta["offsets"] is not None and undo_letterbox:
        oy, ox, nh, nw = meta["offsets"]
        pred_c = pred_mask[oy:oy+nh, ox:ox+nw]
        pred_full = cv2.resize(pred_c, (W0, H0), interpolation=cv2.INTER_NEAREST)
    else:
        pred_full = cv2.resize(pred_mask, (W0, H0), interpolation=cv2.INTER_NEAREST)

    vis_full = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
    return pred_full, vis_full

def save_overlay(path_out: Path, overlay_rgb: np.ndarray):
    """Write overlay RGB as BGR to disk."""
    path_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path_out), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
