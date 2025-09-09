# inference_sam2_semantic.py
import argparse
from pathlib import Path
import torch

from surgseg.config import NUM_CLASSES
from surgseg.models import create_sam2_semantic
from surgseg.utils import (
    parse_size, get_device, load_image_as_tensor, resize_logits_to,
    load_checkpoint_state, undo_letterbox_and_upscale, overlay_mask, save_overlay
)

@torch.no_grad()
def main():
    p = argparse.ArgumentParser("SAM2 semantic head inference")
    p.add_argument("--frame", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--size", default="1024x576")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--save", type=Path, default="predictions")
    p.add_argument("--undo_letterbox", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # SAM2 / SAM v1 fallback (must match training)
    p.add_argument("--sam2_cfg", default=None, type=Path, help="Path to SAM 2 YAML (if used at train time)")
    p.add_argument("--sam2_ckpt", default=None, type=Path, help="Path to SAM 2 checkpoint (if used at train time)")
    p.add_argument("--fallback_variant", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    args = p.parse_args()

    device = get_device()
    model = create_sam2_semantic(
        NUM_CLASSES,
        sam2_cfg=str(args.sam2_cfg) if args.sam2_cfg else None,
        sam2_ckpt=str(args.sam2_ckpt) if args.sam2_ckpt else None,
        freeze_backbone=True,
        fallback_variant=args.fallback_variant
    ).to(device)
    model.eval()

    load_checkpoint_state(model, args.checkpoint, strict=False, verbose=args.verbose)

    target_size = parse_size(args.size) if args.size else None
    x, rgb_proc, meta = load_image_as_tensor(args.frame, target_size)
    x = x.to(device)

    logits = model(pixel_values=x).logits
    logits = resize_logits_to(logits, (rgb_proc.shape[0], rgb_proc.shape[1]))
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype("uint8")

    pred_full, vis_full = undo_letterbox_and_upscale(pred, args.frame, meta, args.undo_letterbox)
    overlay = overlay_mask(vis_full, pred_full, alpha=args.alpha, num_classes=NUM_CLASSES)

    out_path = Path(args.save) / (str(args.frame).replace('/', "-") + "_sam2_masked.png")
    save_overlay(out_path, overlay)
    print(f"[inference] saved -> {out_path}")


if __name__ == "__main__":
    main()
