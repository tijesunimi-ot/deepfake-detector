# src/quant_utils.py
import torch
import torch.nn as nn
from typing import List

def fuse_mobilenetv2(model: nn.Module):
    """
    Fuse Conv-BN-ReLU patterns in torchvision MobileNetV2 for quantization.
    torchvision's mobilenet_v2 uses InvertedResidual blocks with nested modules.
    """
    # top-level first conv + bn + relu6 is already in features[0]
    for m in model.modules():
        if type(m).__name__ == "ConvBNReLU":
            torch.ao.quantization.fuse_modules(m, ["0", "1", "2"], inplace=True)
    # InvertedResidual: fuse inside blocks where applicable
    for idx, m in enumerate(model.features):
        if m.__class__.__name__ == "InvertedResidual":
            # the block has a sequence of ConvBNReLU and a final Conv+BN
            for inner in m.conv:
                if type(inner).__name__ == "ConvBNReLU":
                    torch.ao.quantization.fuse_modules(inner, ["0", "1", "2"], inplace=True)
            # last two of the block: conv (index -2) & bn (index -1) if they exist
            # This is implementation dependent; try-catch to be safe.
            try:
                torch.ao.quantization.fuse_modules(m.conv, ["0.0", "0.1", "0.2"], inplace=True)  # no-op if already fused
            except Exception:
                pass
            try:
                torch.ao.quantization.fuse_modules(m.conv, ["3", "4"], inplace=True)
            except Exception:
                pass
    # classifier is Linear + Dropout; nothing to fuse there
    return model
