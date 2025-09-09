import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large
)

def freeze_bn(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if m.affine:
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

def convert_bn_to_gn(module: nn.Module, num_groups=32):
    for name, m in list(module.named_children()):
        if isinstance(m, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups=min(num_groups, m.num_features), num_channels=m.num_features)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(m, num_groups)
    return module

def create_model(num_classes, backbone="resnet50", enable_aux=True):
    if backbone == "mobilenet":
        model = deeplabv3_mobilenet_v3_large(weights="DEFAULT", aux_loss=True)
        in_ch = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        if model.aux_classifier is not None:
            in_ch_aux = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)
            if not enable_aux:
                model.aux_classifier = None
    else:
        model = deeplabv3_resnet50(weights="DEFAULT", aux_loss=enable_aux)
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        if model.aux_classifier is not None and not enable_aux:
            model.aux_classifier = None
    return model
