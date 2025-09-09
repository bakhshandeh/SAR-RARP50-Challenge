# train_deeplab.py
import os, math, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from surgseg.config import NUM_CLASSES, IGNORE_LABEL, make_class_weights
from surgseg.data import SurgicalSegDataset, load_label_map
from surgseg.losses import CombinedLoss
from surgseg.models import create_model, freeze_bn, convert_bn_to_gn
from surgseg.engine import train_one_epoch, evaluate
from surgseg.utils import set_seed, get_device, save_ckpt, parse_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Dataset root with train/video_*/rgb|segmentation and test/video_*/rgb|segmentation")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--size", type=str, default="1024x576", help="WxH")
    ap.add_argument("--tau_nsd", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--tta", action="store_true", help="Enable flip-TTA during validation")
    ap.add_argument("--out", type=str, default="checkpoints/deeplab")
    ap.add_argument("--seed", type=int, default=1337)

    # Labels / model config
    ap.add_argument("--label_map", type=str, default=None,
                    help="JSON mapping raw labels -> {0..NUM_CLASSES-1 or 255}. If unset, out-of-range -> 255")
    ap.add_argument("--backbone", type=str, default="mobilenet",
                    choices=["resnet50", "mobilenet"], help="DeepLab backbone")
    ap.add_argument("--disable_aux", action="store_true",
                    help="Disable aux head (MobileNet still builds with aux then drops it)")
    ap.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps (>=1)")
    ap.add_argument("--bn", type=str, default="bn",
                    choices=["bn", "freeze", "gn"],
                    help="BatchNorm strategy: normal BN, freeze BN (eval), or GroupNorm")

    # Progress
    ap.add_argument("--progress", type=str, default="bar",
                    choices=["bar", "steps", "none"],
                    help="Progress reporting: tqdm bar, periodic step prints, or none")
    ap.add_argument("--log_every", type=int, default=50,
                    help="Print every N steps when --progress steps")

    args = ap.parse_args()

    target_size = parse_size(args.size)
    set_seed(args.seed)
    device = get_device()

    label_map = load_label_map(args.label_map) if args.label_map else None
    if label_map is not None:
        print(f"Loaded label map with {len(label_map)} entries.")

    # === Datasets ===
    # data.py reads directly from:
    #   train -> root/train/video_*/rgb|segmentation
    #   val   -> root/test/video_*/rgb|segmentation
    train_ds = SurgicalSegDataset(args.root, split="train",
                                  target_size=target_size, use_albu=True,
                                  label_map=label_map)
    val_ds   = SurgicalSegDataset(args.root, split="val",
                                  target_size=target_size, use_albu=True,
                                  label_map=label_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # === Model ===
    model = create_model(NUM_CLASSES, backbone=args.backbone, enable_aux=(not args.disable_aux)).to(device)

    # BN strategy
    if args.bn == "freeze":
        freeze_bn(model)
    elif args.bn == "gn":
        convert_bn_to_gn(model, num_groups=32)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # === Loss / Opt ===
    class_weights = make_class_weights().to(device)
    # Keep boundary_weight=0.0 for DeepLab default (you can raise it if you want more edge emphasis)
    loss_fn = CombinedLoss(class_weights=class_weights,
                           dice_weight=0.25, ce_weight=0.75,
                           boundary_weight=0.0, ignore_index=IGNORE_LABEL)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.out, exist_ok=True)
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler,
            accum_steps=max(1, args.accum_steps),
            progress=args.progress, log_every=args.log_every,
            epoch=epoch, epochs=args.epochs,
            aux_weight=0.4,              # use aux if present (DeepLab)
            deep_lab_dict_out=True,      # DeepLab returns dict with "out"/"aux"
            segformer_style=False
        )
        scheduler.step()

        metrics = evaluate(model, val_loader, device, tau_nsd=args.tau_nsd,
                           show_bar=(args.progress == "bar"),
                           tta=args.tta, segformer_style=False)

        dt = time.time() - t0
        print(f"[{epoch:03d}/{args.epochs}] loss={tr_loss:.4f}  "
              f"mIoU={metrics['mIoU']:.4f}  mNSD={metrics['mNSD']:.4f}  "
              f"S={metrics['score']:.4f}  ({dt:.1f}s)")

        if metrics["score"] > best:
            best = metrics["score"]
            save_ckpt(os.path.join(args.out, "best.pth"), model, optimizer, epoch, best)
        if epoch % 5 == 0:
            save_ckpt(os.path.join(args.out, f"epoch_{epoch}.pth"), model, optimizer, epoch, best)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. Best √(mIoU·mNSD) = {best:.4f}")

if __name__ == "__main__":
    main()
