# surgseg/data.py
import random, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IGNORE_LABEL, NUM_CLASSES, IMG_EXTS, MASK_EXTS

# ----- optional Albumentations -----
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBU = True
except Exception:
    HAS_ALBU = False
    print("Albumentations not found. Using light torchvision transforms.")

# ===== Label utilities =====
def load_label_map(path_json):
    with open(path_json, "r") as f:
        m = json.load(f)
    return {int(k): int(v) for k, v in m.items()}

def apply_label_map(mask_np: np.ndarray, label_map: dict, ignore_val=IGNORE_LABEL):
    out = np.full_like(mask_np, ignore_val, dtype=np.int32)
    for src, dst in label_map.items():
        out[mask_np == src] = dst
    return out

def sanitize_mask_range(mask_np: np.ndarray, num_classes=NUM_CLASSES, ignore_val=IGNORE_LABEL):
    m = mask_np.astype(np.int32)
    ok = (m >= 0) & (m < num_classes)
    return np.where(ok, m, ignore_val).astype(np.int32)

# ===== Albumentations builder =====
def build_albu(split: str, target_size, ignore_val=IGNORE_LABEL):
    if not HAS_ALBU:
        return None
    h, w = target_size[1], target_size[0]
    if split == "train":
        return A.Compose([
            A.RandomScale(scale_limit=(-0.3, 0.2), p=1.0),
            A.LongestMaxSize(max_size=target_size[0], interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT,
                          value=0, mask_value=ignore_val),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=7,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.75),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.2, p=1),
                A.CLAHE(clip_limit=2.0, p=1),
            ], p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.35),
            A.MotionBlur(blur_limit=7, p=0.25),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=target_size[0], interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT,
                          value=0, mask_value=ignore_val),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# ===== Dataset =====
class SurgicalSegDataset(Dataset):
    """
    Dataset layout:
      root/
        train/
          video_XXXX/
            rgb/*.png
            segmentation/*.png
        test/
          video_YYYY/
            *.png (same rules)
    """
    def __init__(self, root, split="train",
                 target_size=(1024, 576), use_albu=True,
                 train_ids=None, val_ids=None,
                 label_map=None):
        self.root = Path(root)
        self.target_size = target_size
        self.split = split
        self.use_albu = use_albu and HAS_ALBU
        self.label_map = label_map

        # print("root, split", root, split)
        split_dir = self._resolve_split_dir(self.root, split)
        # print("split_dir", split_dir)
        all_pairs = self._find_pairs(split_dir)

        if train_ids is not None and val_ids is not None:
            keep = set(train_ids if split == "train" else val_ids)
            self.samples = [p for p in all_pairs if p["rel_id"] in keep]
        else:
            self.samples = all_pairs

        if self.use_albu:
            self.transform = build_albu(self.split, self.target_size, IGNORE_LABEL)
        else:
            self.transform = None
            self.to_tensor = transforms.ToTensor()
            self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                  std=(0.229, 0.224, 0.225))
        print(f"[{split}] samples: {len(self.samples)}")

    @staticmethod
    def _resolve_split_dir(root: Path, split: str) -> Path:
        if split == "train":
            cand = root / "train"
        else:
            cand = root / "test"
        # print("cand", cand)
        return cand if cand.exists() and cand.is_dir() else root

    @staticmethod
    def _find_pairs(scan_root: Path):
        """
        Build (img, mask) pairs strictly from:
            scan_root/video_*/rgb/*.png
            scan_root/video_*/segmentation/*.png

        Each pair is matched by shared stem name within the same video_* directory.
        rel_id format: "video_xxxx/<stem>"
        """
        pairs = []
        video_dirs = sorted([d for d in scan_root.iterdir() if d.is_dir() and d.name.startswith("video_")])

        for vd in video_dirs:
            rgb_dir = vd / "rgb"
            seg_dir = vd / "segmentation"
            if not (rgb_dir.exists() and seg_dir.exists()):
                # Skip directories that do not contain both subfolders
                continue

            # index masks by stem
            mask_map = {}
            for m in seg_dir.iterdir():
                if m.is_file() and m.suffix.lower() in MASK_EXTS:
                    mask_map[m.stem] = m

            # pair images to masks by stem
            for im in rgb_dir.iterdir():
                if not (im.is_file() and im.suffix.lower() in IMG_EXTS):
                    continue
                stem = im.stem
                m = mask_map.get(stem, None)
                if m is None:
                    continue
                rel_id = f"{vd.name}/{stem}"
                pairs.append({"rel_id": rel_id, "img_path": im, "mask_path": m})

        if not pairs:
            raise RuntimeError(
                f"No (image, mask) pairs found under {scan_root}.\n"
                "Expected layout:\n"
                "  <root>/{train|test}/video_*/rgb/*.png\n"
                "  <root>/{train|test}/video_*/segmentation/*.png\n"
                "with matching filenames (same stem) in rgb/ and segmentation/."
            )
        return sorted(pairs, key=lambda d: d["rel_id"])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        itm = self.samples[idx]
        img = cv2.imread(str(itm["img_path"]))[:, :, ::-1]
        mask = cv2.imread(str(itm["mask_path"]), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(itm["mask_path"])
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.int32)

        if self.label_map is not None:
            mask = apply_label_map(mask, self.label_map, ignore_val=IGNORE_LABEL)
        else:
            mask = sanitize_mask_range(mask, NUM_CLASSES, ignore_val=IGNORE_LABEL)

        if self.use_albu:
            r = self.transform(image=img, mask=mask)
            img_t = r["image"]
            mask_t = r["mask"]
            mask_t = (mask_t.squeeze() if isinstance(mask_t, torch.Tensor) else np.squeeze(mask_t))
            mask_t = torch.as_tensor(mask_t, dtype=torch.long)
        else:
            target_w, target_h = self.target_size
            h, w = img.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_rs = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_rs = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            canvas_mask = np.full((target_h, target_w), IGNORE_LABEL, dtype=np.int32)
            off_x = (target_w - new_w) // 2
            off_y = (target_h - new_h) // 2
            canvas[off_y:off_y+new_h, off_x:off_x+new_w] = img_rs
            canvas_mask[off_y:off_y+new_h, off_x:off_x+new_w] = mask_rs
            img_t = transforms.ToTensor()(Image.fromarray(canvas))
            img_t = transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))(img_t)
            mask_t = torch.from_numpy(canvas_mask).squeeze().long()

        return img_t, mask_t, itm["rel_id"]
