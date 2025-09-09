import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .metrics import compute_confusion_matrix, miou_from_cm, mnsd
from .config import NUM_CLASSES

# ----- evaluation -----
@torch.no_grad()
def evaluate(model, loader, device, tau_nsd=3, show_bar=False, tta=False,
             segformer_style=False):
    model.eval()
    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    nsd_vals = []
    it = tqdm(loader, total=len(loader), desc="val", unit="img", leave=False) if show_bar else loader

    def _infer_logits(imgs):
        if segformer_style:
            return model(pixel_values=imgs).logits
        out = model(imgs)
        return out["out"] if isinstance(out, dict) and "out" in out else out.logits

    for imgs, masks, _ in it:
        imgs = imgs.to(device, non_blocking=True)

        if not tta:
            logits = _infer_logits(imgs)
        else:
            # flip-only TTA (works for all models); segformer has a separate heavy TTA in scripts if desired
            logits_a = _infer_logits(imgs)
            imgs_fl = torch.flip(imgs, dims=[-1])
            logits_b = torch.flip(_infer_logits(imgs_fl), dims=[-1])
            logits = 0.5 * (logits_a + logits_b)

        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1).cpu()
        for p, t in zip(preds, masks):
            cm_total += compute_confusion_matrix(p, t, NUM_CLASSES)
            nsd_frame, _ = mnsd(p.numpy().astype(np.int32),
                                t.numpy().astype(np.int32),
                                NUM_CLASSES, tau=tau_nsd)
            nsd_vals.append(nsd_frame)
    mIoU, _ = miou_from_cm(cm_total)
    mNSD = float(np.mean(nsd_vals)) if nsd_vals else 0.0
    score = math.sqrt(max(mIoU, 0.0) * max(mNSD, 0.0))
    return {"mIoU": mIoU, "mNSD": mNSD, "score": score}

# ----- train loop (generic, supports aux via hook) -----
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None,
                    accum_steps=1, progress="bar", log_every=50, epoch=1, epochs=1,
                    aux_weight=0.0, deep_lab_dict_out=True, segformer_style=False):
    model.train()
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    step_in_accum = 0
    iterator = loader if progress != "bar" else tqdm(loader, total=len(loader), desc=f"train {epoch}/{epochs}", unit="batch", leave=False)

    for step, (imgs, masks, _) in enumerate(iterator):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        use_autocast = (scaler is not None)
        # torch>=2 supports torch.amp.autocast; keep cuda autocast variant for compatibility
        ctx = torch.cuda.amp.autocast(enabled=use_autocast) if torch.cuda.is_available() else torch.cpu.amp.autocast(enabled=use_autocast)
        with ctx:
            if segformer_style:
                out = model(pixel_values=imgs)
                logits = out.logits
            else:
                out = model(imgs)
                if deep_lab_dict_out and isinstance(out, dict):
                    logits = out["out"]
                else:
                    logits = out.logits
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss_fn(logits, masks)

            if deep_lab_dict_out and isinstance(out, dict) and ("aux" in out) and (out["aux"] is not None):
                loss = loss + aux_weight * loss_fn(out["aux"], masks)

            loss = loss / max(1, accum_steps)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        step_in_accum += 1
        if step_in_accum == max(1, accum_steps):
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0

        total += loss.item() * max(1, accum_steps)
        if progress == "bar":
            iterator.set_postfix(loss=f"{loss.item()*max(1,accum_steps):.4f}")
        elif progress == "steps" and (step % max(1, log_every) == 0):
            print(f"  step {step:5d}/{len(loader)}  loss={loss.item()*max(1,accum_steps):.4f}", flush=True)

    if step_in_accum > 0:
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total / max(1, len(loader))

# ----- TTA helpers -----
@torch.no_grad()
def tta_multiscale_flip_segformer(model, imgs, out_size, device, scales=(0.75, 1.0, 1.25, 1.5), hflip=True):
    model.eval()
    H, W = out_size
    sum_logits = None
    with torch.no_grad():
        for s in scales:
            nh = int(round(H * s)); nw = int(round(W * s))
            im = torch.nn.functional.interpolate(imgs, size=(nh, nw), mode="bilinear", align_corners=False)
            out = model(pixel_values=im.to(device)).logits
            out = torch.nn.functional.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
            sum_logits = out if sum_logits is None else (sum_logits + out)
            if hflip:
                imf = torch.flip(im, dims=[-1])
                outf = model(pixel_values=imf.to(device)).logits
                outf = torch.flip(outf, dims=[-1])
                outf = torch.nn.functional.interpolate(outf, size=(H, W), mode="bilinear", align_corners=False)
                sum_logits += outf
    denom = float(len(scales) * (2 if hflip else 1))
    return sum_logits / denom
