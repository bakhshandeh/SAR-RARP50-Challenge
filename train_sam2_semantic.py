# train_sam2_semantic.py
import os, argparse, time
import torch
from torch.utils.data import DataLoader

from surgseg.config import NUM_CLASSES, IGNORE_LABEL, make_class_weights
from surgseg.data import SurgicalSegDataset, load_label_map
from surgseg.losses import CombinedLoss
from surgseg.models import create_sam2_semantic
from surgseg.engine import train_one_epoch, evaluate
from surgseg.utils import set_seed, get_device, save_ckpt, parse_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Dataset root with train/video_*/{rgb,segmentation} and test/video_*/{rgb,segmentation}")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--size", type=str, default="1024x576", help="WxH")
    ap.add_argument("--tau_nsd", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--progress", type=str, default="bar", choices=["bar","steps","none"])
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", type=str, default="checkpoints/sam2_semantic")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--label_map", type=str, default=None)

    # SAM2/SAM-v1 options
    ap.add_argument("--sam2_cfg", type=str, default=None, help="Path to SAM 2 YAML (use SAM2 if set with --sam2_ckpt)")
    ap.add_argument("--sam2_ckpt", type=str, default=None, help="Path to SAM 2 checkpoint")
    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze image encoder weights")
    ap.add_argument("--fallback_variant", type=str, default="vit_h",
                    choices=["vit_h","vit_l","vit_b"], help="SAM v1 fallback if SAM2 not provided")

    # Loss options
    ap.add_argument("--dice_w", type=float, default=0.25)
    ap.add_argument("--ce_w", type=float, default=0.70)
    ap.add_argument("--boundary_w", type=float, default=0.05)
    ap.add_argument("--boundary_radius", type=int, default=1)

    # Eval options
    ap.add_argument("--tta", action="store_true", help="Enable flip TTA at validation")

    args = ap.parse_args()

    target_size = parse_size(args.size)
    set_seed(args.seed)
    device = get_device()

    label_map = load_label_map(args.label_map) if args.label_map else None
    if label_map is not None:
        print(f"Loaded label map with {len(label_map)} entries.")

    # === Datasets ===
    # data reads directly from:
    #   split="train" -> <root>/train/video_*/{rgb,segmentation}
    #   split={"val","test"} -> <root>/test/video_*/{rgb,segmentation}
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
    model = create_sam2_semantic(
        NUM_CLASSES,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        freeze_backbone=args.freeze_backbone,
        fallback_variant=args.fallback_variant
    ).to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # === Loss / Opt ===
    class_weights = make_class_weights().to(device)
    loss_fn = CombinedLoss(class_weights=class_weights,
                           dice_weight=args.dice_w, ce_weight=args.ce_w,
                           boundary_weight=args.boundary_w, boundary_radius=args.boundary_radius,
                           ignore_index=IGNORE_LABEL)

    # Separate lr for head/backbone if backbone is unfrozen
    head_params = [p for p in model.decode_head.parameters() if p.requires_grad]
    back_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if len(back_params) == 0:
        optimizer = torch.optim.AdamW([{"params": head_params, "lr": args.lr}],
                                      lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": args.lr},
            {"params": back_params, "lr": args.lr * 0.1}
        ], weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(args.out, exist_ok=True)
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler,
            accum_steps=max(1, args.accum_steps),
            progress=args.progress, log_every=args.log_every,
            epoch=epoch, epochs=args.epochs,
            aux_weight=0.0,               # head-only; no aux branch here
            deep_lab_dict_out=False,      # model returns an object with .logits
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
            save_ckpt(os.path.join(args.out, "best_sam2_semantic.pth"), model, optimizer, epoch, best)
        if epoch % 5 == 0:
            save_ckpt(os.path.join(args.out, f"sam2_semantic_epoch_{epoch}.pth"), model, optimizer, epoch, best)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. Best √(mIoU·mNSD) = {best:.4f}")

if __name__ == "__main__":
    main()
