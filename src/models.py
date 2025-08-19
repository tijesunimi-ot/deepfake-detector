# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Attempt to use torchvision + timm
try:
    import torchvision.models as tv_models
except Exception:
    tv_models = None
try:
    import timm
except Exception:
    timm = None

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ClassifierHead(nn.Module):
    """
    Simple classifier head: global avgpool -> dropout -> fc
    Works for backbones that output feature maps or 1D features.
    """
    def __init__(self, in_features, num_classes=2, p_drop=0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) if in_features is not None else None
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # if x is a feature map: (B, C, H, W)
        if x.dim() == 4:
            x = self.pool(x)
            x = x.view(x.size(0), -1)
        # else assume already (B, C)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def get_mobilenet_v2(num_classes: int = 2,
                     pretrained: bool = True,
                     input_size: int = 160,
                     p_drop: float = 0.2,
                     freeze_backbone: bool = False):
    """
    Return a MobileNetV2 adapted for arbitrary input_size (e.g., 160).
    Notes:
    - MobileNetV2 is fully-convolutional so it accepts different spatial sizes
      (160x160 is supported). Pretrained weights were trained at 224x224; using
      a smaller input is fine but may slightly change performance.
    - Ensure your data pipeline resizes crops to `input_size` before feeding the model.
    - We keep the standard MobileNetV2 last_channel (usually 1280) and replace
      the classifier head to match num_classes.
    """
    if tv_models is None:
        raise RuntimeError("torchvision is required for MobileNetV2. Install torchvision.")
    # load pretrained backbone
    mobilenet = tv_models.mobilenet_v2(pretrained=pretrained)
    # mobilenet.features outputs feature map; classifier is mobilenet.classifier
    # we'll replace classifier with our smaller head (to make distillation easier)
    # find number of features:
    last_channel = mobilenet.last_channel if hasattr(mobilenet, "last_channel") else 1280
    # Replace classifier
    mobilenet.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(last_channel, num_classes)
    )
    if freeze_backbone:
        for name, p in mobilenet.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False
    return mobilenet

def get_xception(num_classes: int = 2, pretrained: bool = True):
    """
    Returns an Xception model as a teacher. Uses timm if available.
    """
    if timm is None:
        raise RuntimeError("timm is required for Xception. Install timm (`pip install timm`).")
    # timm has xception variants (e.g., 'xception', 'xception41' depending on version)
    model_name_candidates = ["xception", "xception41", "xception_tf"]
    found = None
    for name in model_name_candidates:
        if name in timm.list_models(pretrained=False):
            found = name
            break
    if found is None:
        # fallback: use a strong alternative such as 'resnet50d'
        print("Xception not found in timm model list; falling back to resnet50d as teacher.")
        found = "resnet50d"
    teacher = timm.create_model(found, pretrained=pretrained, num_classes=num_classes)
    return teacher

class FeatureExtractorWrapper(nn.Module):
    """
    Wrap a backbone to expose intermediate features if needed.
    For most backbones, we simply return logits. This wrapper can be extended.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

class DistillationLoss(nn.Module):
    """
    Distillation loss: alpha * CE(student, labels) + (1-alpha) * KL(soft(student/T), soft(teacher/T)) * T^2
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5, reduction="batchmean"):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction=reduction)

    def forward(self, student_logits, teacher_logits, labels):
        """
        student_logits: (B, C) raw logits
        teacher_logits: (B, C) raw logits (teacher outputs)
        labels: (B,) long
        """
        # classification loss
        loss_ce = self.ce(student_logits, labels)
        # soft targets
        s_log_prob = F.log_softmax(student_logits / self.T, dim=1)
        t_prob = F.softmax(teacher_logits / self.T, dim=1)
        loss_kl = self.kl(s_log_prob, t_prob) * (self.T ** 2)
        loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kl
        return loss, loss_ce.detach(), loss_kl.detach()

# Simple factory
def get_model(name: str = "mobilenet_v2", num_classes: int = 2, pretrained: bool = True, freeze_backbone=False):
    name = name.lower()
    if name in ["mobilenet", "mobilenet_v2", "mobilenetv2", "mobile_net_v2"]:
        model = get_mobilenet_v2(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif name in ["xception", "teacher_xception", "teacher"]:
        model = get_xception(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model

if __name__ == "__main__":
    # quick sanity checks
    device = get_device()
    print("Device:", device)
    try:
        student = get_model("mobilenet_v2", num_classes=2, pretrained=True)
        print("MobileNetV2 params (trainable):", count_parameters(student))
    except Exception as e:
        print("Failed to create MobileNetV2:", e)
    try:
        teacher = get_model("xception", num_classes=2, pretrained=True)
        print("Teacher (Xception/resnet fallback) params (trainable):", count_parameters(teacher))
    except Exception as e:
        print("Failed to create teacher model:", e)
        print("Ensure you have torchvision and timm installed for model creation.")