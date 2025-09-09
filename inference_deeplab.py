# inference_deeplab.py
import argparse
from pathlib import Path
import torch

from surgseg.config import NUM_CLASSES
from surgseg.models import create_model
from surgseg.utils import (
    parse_size, get_device, load_image_as_tensor, resize_logits_to,
    load_checkpoint_state, undo_letterbox_and_upscale, overlay_mask, save_overlay
)

@torch.no_grad()
def main():
    p = argparse.ArgumentParser("DeepLabV3 inference")
    p.add_argument("--frame", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--backbone", default="mobilenet", choices=["mobilenet","resnet50"])
    p.add_argument("--size", default="1024x576")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--save", type=Path, default="predictions")
    p.add_argument("--undo_letterbox", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = get_device()
    model = create_model(NUM_CLASSES, backbone=args.backbone, enable_aux=False).to(device)
    model.eval()

    load_checkpoint_state(model, args.checkpoint, strict=False, verbose=args.verbose)

    target_size = parse_size(args.size) if args.size else None
    x, rgb_proc, meta = load_image_as_tensor(args.frame, target_size)
    x = x.to(device)

    out = model(x)
    logits = out["out"] if isinstance(out, dict) else out
    logits = resize_logits_to(logits, (rgb_proc.shape[0], rgb_proc.shape[1]))
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype("uint8")

    pred_full, vis_full = undo_letterbox_and_upscale(pred, args.frame, meta, args.undo_letterbox)
    overlay = overlay_mask(vis_full, pred_full, alpha=args.alpha, num_classes=NUM_CLASSES)

    out_path = Path(args.save) / (str(args.frame).replace('/', "-") + "_deeplab_masked.png")
    save_overlay(out_path, overlay)
    print(f"[inference] saved -> {out_path}")



if __name__ == "__main__":
    main()
