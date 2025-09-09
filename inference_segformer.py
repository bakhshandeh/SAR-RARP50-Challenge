# inference_segformer.py
import argparse
from pathlib import Path
import torch

from surgseg.config import NUM_CLASSES, IGNORE_LABEL
from surgseg.models import create_segformer
from surgseg.utils import (
    parse_size, get_device, load_image_as_tensor, resize_logits_to,
    load_checkpoint_state, undo_letterbox_and_upscale, overlay_mask, save_overlay
)

@torch.no_grad()
def main():
    p = argparse.ArgumentParser("SegFormer inference")
    p.add_argument("--frame", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--variant", default="b2", choices=["b0","b1","b2","b3","b4","b5"])
    p.add_argument("--size", default="1024x576", help="WxH canvas (use training size)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--save", type=Path, default="predictions")
    p.add_argument("--undo_letterbox", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = get_device()
    model = create_segformer(NUM_CLASSES, variant=args.variant, ignore_index=IGNORE_LABEL).to(device)
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

    out_path = Path(args.save) / (str(args.frame).replace('/', "-") + "_segformer_masked.png")
    save_overlay(out_path, overlay)
    print(f"[inference] saved -> {out_path}")

if __name__ == "__main__":
    main()
