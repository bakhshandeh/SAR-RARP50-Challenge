import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional SAM2 / SAM-v1 imports
_HAS_SAM2 = False
_HAS_SAM = False
sam2_build_fn = None
sam2_cfg_loader = None
try:
    from sam2.build_sam import build_sam2
    from sam2.utils.config import load_config as sam2_load_config
    sam2_build_fn = build_sam2
    sam2_cfg_loader = sam2_load_config
    _HAS_SAM2 = True
except Exception:
    pass
try:
    from segment_anything import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
    _HAS_SAM = True
except Exception:
    pass

_SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1,3,1,1)
_SAM_PIXEL_STD  = torch.tensor([58.395, 57.12, 57.375]).view(1,3,1,1)

def _denorm_imagenet(x: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x * std) + mean

def _resize_pad_to_1024(x: torch.Tensor, size=1024):
    b, c, h, w = x.shape
    scale = size / max(h, w)
    new_h = int(round(h * scale / 16)) * 16
    new_w = int(round(w * scale / 16)) * 16
    new_h = min(new_h, size); new_w = min(new_w, size)
    x_rs = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_t = (size - new_h) // 2
    pad_b = size - new_h - pad_t
    pad_l = (size - new_w) // 2
    pad_r = size - new_w - pad_l
    x_pad = F.pad(x_rs, (pad_l, pad_r, pad_t, pad_b))
    return x_pad

class TinyFPNHead(nn.Module):
    def __init__(self, num_classes, mid_ch=256):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.LazyConv2d(mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        self.aspp_branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 1, bias=False),
                          nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 3, padding=6, dilation=6, bias=False),
                          nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 3, padding=12, dilation=12, bias=False),
                          nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 3, padding=18, dilation=18, bias=False),
                          nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)),
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch * (len(self.aspp_branches) + 1), mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, num_classes, 1)
        )
    def forward(self, x):
        x = self.reduce(x)
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.aspp_branches]
        img = self.image_pool(x)
        img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(img)
        x = torch.cat(feats, dim=1)
        x = self.project(x)
        x = self.refine(x)
        return x

class SAM2SemanticSeg(nn.Module):
    def __init__(self,
                 num_classes: int,
                 sam2_cfg_path: str = None,
                 sam2_ckpt_path: str = None,
                 sam_variant_fallback: str = "vit_h",
                 freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.is_sam2 = False
        if _HAS_SAM2 and (sam2_cfg_path and sam2_ckpt_path):
            cfg = sam2_cfg_loader(sam2_cfg_path)
            model = sam2_build_fn(cfg, sam2_ckpt_path, device="cpu")
            self.backbone = model.image_encoder
            self.is_sam2 = True
        elif _HAS_SAM:
            if sam_variant_fallback == "vit_l":
                self.backbone = build_sam_vit_l(checkpoint=None).image_encoder
            elif sam_variant_fallback == "vit_b":
                self.backbone = build_sam_vit_b(checkpoint=None).image_encoder
            else:
                self.backbone = build_sam_vit_h(checkpoint=None).image_encoder
        else:
            raise ImportError(
                "Install SAM 2 or SAM v1: "
                "pip install 'git+https://github.com/facebookresearch/segment-anything-2.git' "
                "or 'git+https://github.com/facebookresearch/segment-anything.git'"
            )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.decode_head = TinyFPNHead(num_classes=num_classes)

        class _Out:
            def __init__(self, logits): self.logits = logits
        self._Out = _Out

    @torch.inference_mode()
    def _encode_sam2(self, x: torch.Tensor):
        feats = self.backbone(x)
        if isinstance(feats, dict):
            cand = None
            for v in feats.values():
                if cand is None or v.shape[-1]*v.shape[-2] < cand.shape[-1]*cand.shape[-2]:
                    cand = v
            return cand
        elif isinstance(feats, (list, tuple)):
            return feats[-1]
        else:
            return feats

    @torch.inference_mode()
    def _encode_sam(self, x: torch.Tensor):
        x = _denorm_imagenet(x)
        x = _resize_pad_to_1024(x, 1024)
        mean = _SAM_PIXEL_MEAN.to(x.device)
        std  = _SAM_PIXEL_STD.to(x.device)
        x = (x * 255.0 - mean) / std
        return self.backbone(x)

    def forward(self, pixel_values: torch.Tensor):
        if self.is_sam2:
            with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
                f = self._encode_sam2(pixel_values)
        else:
            with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
                f = self._encode_sam(pixel_values)
        logits = self.decode_head(f)
        class _Out:
            def __init__(self, logits): self.logits = logits
        return _Out(logits)

def create_sam2_semantic(num_classes: int,
                         sam2_cfg: str = None,
                         sam2_ckpt: str = None,
                         freeze_backbone: bool = True,
                         fallback_variant: str = "vit_h"):
    return SAM2SemanticSeg(
        num_classes=num_classes,
        sam2_cfg_path=sam2_cfg,
        sam2_ckpt_path=sam2_ckpt,
        sam_variant_fallback=fallback_variant,
        freeze_backbone=freeze_backbone
    )
