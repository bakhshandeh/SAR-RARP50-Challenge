# train_segformer.py
import os, math, argparse, time, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from surgseg.config import NUM_CLASSES, IGNORE_LABEL, make_class_weights
from surgseg.data import SurgicalSegDataset, load_label_map
from surgseg.losses import CombinedLoss
from surgseg.models import create_segformer
from surgseg.engine import train_one_epoch, evaluate, tta_multiscale_flip_segformer
from surgseg.utils import set_seed, get_device, save_ckpt, parse_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Dataset root with train/video_*/*.png and test/video_*/*.png")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--size", type=str, default="1024x576", help="WxH")
    ap.add_argument("--tau_nsd", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--variant", type=str, default="b2",
                    choices=["b0","b1","b2","b3","b4","b5"], help="SegFormer size")
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--progress", type=str, default="bar", choices=["bar","steps","none"])
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--label_map", type=str, default=None,
                    help="JSON raw→{0..K-1 or 255}; if unset, out-of-range→255")
    ap.add_argument("--tta", action="store_true",
                    help="Enable multi-scale+flip TTA at validation")
    args = ap.parse_args()

    target_size = parse_size(args.size)
    set_seed(args.seed)
    device = get_device()

    label_map = load_label_map(args.label_map) if args.label_map else None
    if label_map is not None:
        print(f"Loaded label map with {len(label_map)} entries.")

    # Datasets auto-read from root/train/... and root/test/...
    train_ds = SurgicalSegDataset(args.root, split="train",
                                  target_size=target_size, use_albu=True,
                                  label_map=label_map)
    val_ds   = SurgicalSegDataset(args.root, split="val",   # "val" maps to root/test
                                  target_size=target_size, use_albu=True,
                                  label_map=label_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---- Model ----
    model = create_segformer(NUM_CLASSES, variant=args.variant, ignore_index=IGNORE_LABEL).to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Loss / Opt ----
    class_weights = make_class_weights().to(device)
    # keep core weights the same; add a light boundary term as before
    loss_fn = CombinedLoss(class_weights=class_weights,
                           dice_weight=0.25, ce_weight=0.75,
                           boundary_weight=0.05, boundary_radius=1,
                           ignore_index=IGNORE_LABEL)
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
            aux_weight=0.0,            # no aux head in SegFormer
            deep_lab_dict_out=False,   # model returns .logits
            segformer_style=True
        )
        scheduler.step()

        # Validation (with optional TTA)
        if not args.tta:
            metrics = evaluate(model, val_loader, device,
                               tau_nsd=args.tau_nsd,
                               show_bar=(args.progress == "bar"),
                               tta=False,
                               segformer_style=True)
        else:
            # replicate original multiscale+flip TTA path
            model.eval()
            from surgseg.metrics import compute_confusion_matrix, miou_from_cm, mnsd
            cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            nsd_vals = []
            with torch.no_grad():
                it = tqdm(val_loader, total=len(val_loader), desc="val+tta", unit="img", leave=False) \
                     if args.progress == "bar" else val_loader
                for imgs, masks, _ in it:
                    imgs = imgs.to(device, non_blocking=True)
                    logits = tta_multiscale_flip_segformer(
                        model, imgs, masks.shape[-2:], device,
                        scales=(0.75, 1.0, 1.25, 1.5), hflip=True
                    )
                    preds = torch.argmax(logits, dim=1).cpu()
                    for p, t in zip(preds, masks):
                        cm_total += compute_confusion_matrix(p, t, NUM_CLASSES)
                        nsd_frame, _ = mnsd(p.numpy().astype(np.int32),
                                            t.numpy().astype(np.int32),
                                            NUM_CLASSES, tau=args.tau_nsd)
                        nsd_vals.append(nsd_frame)
            mIoU, _ = miou_from_cm(cm_total)
            mNSD = float(np.mean(nsd_vals)) if nsd_vals else 0.0
            metrics = {"mIoU": mIoU, "mNSD": mNSD, "score": math.sqrt(max(mIoU,0.0)*max(mNSD,0.0))}

        dt = time.time() - t0
        print(f"[{epoch:03d}/{args.epochs}] loss={tr_loss:.4f}  "
              f"mIoU={metrics['mIoU']:.4f}  mNSD={metrics['mNSD']:.4f}  "
              f"S={metrics['score']:.4f}  ({dt:.1f}s)")

        # Save
        if metrics["score"] > best:
            best = metrics["score"]
            save_ckpt(os.path.join(args.out, "best_segformer.pth"), model, optimizer, epoch, best)
        if epoch % 5 == 0:
            save_ckpt(os.path.join(args.out, f"segformer_epoch_{epoch}.pth"), model, optimizer, epoch, best)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. Best √(mIoU·mNSD) = {best:.4f}")

if __name__ == "__main__":
    main()
