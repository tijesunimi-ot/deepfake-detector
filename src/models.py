# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

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
    # timm has xception variants (e.g., 'xception', 'xception41', 'xception_tf' depending on version)
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

# -----------------------------
# Feature distillation scaffold
# -----------------------------
class FeatureHook:
    """
    Minimal forward-hook helper that stores the module's forward output in `.features`.
    Usage:
      hook = FeatureHook(module)
      ... run a forward pass ...
      feats = hook.features   # a torch.Tensor
      hook.close()            # remove hook when done
    """
    def __init__(self, module: nn.Module):
        self.module = module
        self.features = None
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # store a detached copy to avoid adding to student's computation graph
        # keep dtype/device for later matching
        self.features = output.detach()

    def close(self):
        self.hook.remove()

def feature_distillation_loss(student_feats: List[torch.Tensor],
                              teacher_feats: List[torch.Tensor],
                              reduction: str = 'mean') -> torch.Tensor:
    """
    Compute a simple L2 feature distillation loss between lists of feature tensors.
    If spatial shapes differ, both tensors are global-pooled to (B, C) before MSE.
    Args:
      student_feats: list of tensors from student (each tensor BxCxHxW or BxC)
      teacher_feats: list of tensors from teacher (same length as student_feats)
      reduction: 'mean' or 'sum' passed to mse_loss
    Returns:
      scalar tensor (MSE loss)
    """
    if len(student_feats) != len(teacher_feats):
        raise ValueError("student_feats and teacher_feats must have same length")
    total_loss = 0.0
    for s, t in zip(student_feats, teacher_feats):
        if s is None or t is None:
            # skip missing features silently (you may want to warn)
            continue
        # if spatial dims differ, reduce to (B, C) via global average pooling
        if s.dim() == 4 and t.dim() == 4 and (s.shape[2:] != t.shape[2:]):
            s_p = s.mean(dim=[2, 3])  # B x C
            t_p = t.mean(dim=[2, 3])
            total_loss = total_loss + F.mse_loss(s_p, t_p, reduction=reduction)
        elif s.dim() == 4 and t.dim() == 4 and (s.shape == t.shape):
            total_loss = total_loss + F.mse_loss(s, t, reduction=reduction)
        else:
            # handle 2D features or mismatched dims by pooling to (B, C)
            if s.dim() == 2 and t.dim() == 2:
                total_loss = total_loss + F.mse_loss(s, t, reduction=reduction)
            else:
                # fallback: global pool both to (B, C)
                s_p = s.mean(dim=list(range(2, s.dim()))) if s.dim() > 1 else s
                t_p = t.mean(dim=list(range(2, t.dim()))) if t.dim() > 1 else t
                total_loss = total_loss + F.mse_loss(s_p, t_p, reduction=reduction)
    return total_loss

def register_hooks_by_name(model: nn.Module, module_names: List[str]) -> List[FeatureHook]:
    """
    Utility to register FeatureHook objects given attribute-like string paths.
    Example module_names: ["features.4", "features.7"] for torchvision MobileNetV2.
    Returns list of FeatureHook objects in the same order.
    """
    hooks = []
    for name in module_names:
        # navigate attributes
        parts = name.split('.')
        mod = model
        try:
            for p in parts:
                if p.isdigit():
                    mod = mod[int(p)]
                else:
                    mod = getattr(mod, p)
        except Exception as e:
            raise ValueError(f"Failed to find module path '{name}' in model: {e}")
        hooks.append(FeatureHook(mod))
    return hooks

# Add this to src/models.py after FeatureHook / register_hooks_by_name

from typing import Dict, Set, Tuple

def auto_register_matching_hooks(
    student: nn.Module,
    teacher: nn.Module,
    dummy_input: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    max_pairs: int = 3,
    layer_types: Tuple = (nn.Conv2d,),
    input_shape: Tuple[int,int,int] = (3,160,160),
) -> Tuple[List[FeatureHook], List[FeatureHook], List[Tuple[str,str]]]:
    """
    Automatically find and register matching FeatureHook pairs between student and teacher.
    Returns (student_hooks, teacher_hooks, matched_name_pairs) where matched_name_pairs is a list
    of (student_module_name, teacher_module_name).

    Strategy:
      1. Register hooks on all modules of types in `layer_types`.
      2. Run forward on dummy input to collect feature tensors.
      3. Greedily pair student -> teacher layers that have identical spatial dims (HxW),
         and then smallest absolute channel difference.
      4. Return persistent FeatureHook objects for the selected pairs and remove other hooks.

    Notes:
      - This is heuristic — you may want to override the selected names manually for best results.
      - Keep max_pairs small (2-5) to avoid large feature-loss cost in training.
    """
    device = device or get_device()
    student = student.to(device)
    teacher = teacher.to(device)

    # prepare dummy input
    if dummy_input is None:
        b = 1
        c, h, w = input_shape
        dummy_input = torch.randn(b, c, h, w).to(device)
    else:
        dummy_input = dummy_input.to(device)

    # collect candidate modules
    def collect_candidates(model):
        cand = []
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                cand.append((name, module))
        return cand

    student_cands = collect_candidates(student)
    teacher_cands = collect_candidates(teacher)

    if len(student_cands) == 0 or len(teacher_cands) == 0:
        raise RuntimeError("No candidate modules found for provided layer_types. "
                           "Try including Conv2d or another layer type.")

    # register hooks for all candidates (temporary FeatureHook objects)
    student_hooks_all: Dict[str, FeatureHook] = {}
    teacher_hooks_all: Dict[str, FeatureHook] = {}
    for name, mod in student_cands:
        student_hooks_all[name] = FeatureHook(mod)
    for name, mod in teacher_cands:
        teacher_hooks_all[name] = FeatureHook(mod)

    # run forward passes to populate hook.features
    with torch.no_grad():
        try:
            _ = teacher(dummy_input)
        except Exception:
            # some teachers may expect different input size; still try student's forward first
            pass
        _ = student(dummy_input)
        # Try teacher forward again if it failed above
        with torch.no_grad():
            try:
                _ = teacher(dummy_input)
            except Exception:
                # ignore; hooks will be None for teacher if forward failed
                pass

    # gather feature metadata
    def collect_feature_info(hooks_dict):
        info = []
        for name, hook in hooks_dict.items():
            f = hook.features
            if f is None:
                continue
            # Expect BxCxHxW or BxC
            if f.dim() == 4:
                _, ch, hh, ww = f.shape
                info.append((name, ch, (hh, ww)))
            elif f.dim() == 2:
                _, ch = f.shape
                info.append((name, ch, (None, None)))
            else:
                # fallback pooling dims into (None,None)
                info.append((name, f.shape[1] if f.dim() > 1 else f.shape[0], (None, None)))
        return info

    s_info = collect_feature_info(student_hooks_all)
    t_info = collect_feature_info(teacher_hooks_all)

    if len(s_info) == 0 or len(t_info) == 0:
        # cleanup any hooks we created and raise
        for h in student_hooks_all.values(): h.close()
        for h in teacher_hooks_all.values(): h.close()
        raise RuntimeError("Could not collect features from hooks. Try using a different dummy_input size.")

    # greedy pairing: prefer identical spatial dims, then minimal channel difference
    matched_pairs: List[Tuple[str, str]] = []
    used_teacher: Set[str] = set()

    # sort students by spatial area descending (prefer deeper features)
    s_info_sorted = sorted(s_info, key=lambda x: (x[2][0] or 0) * (x[2][1] or 0), reverse=True)

    for s_name, s_ch, s_sp in s_info_sorted:
        best_t = None
        best_score = None
        for t_name, t_ch, t_sp in t_info:
            if t_name in used_teacher:
                continue
            # spatial match score: 0 if equal, 1 if either None or not equal
            spatial_mismatch = 0 if (s_sp == t_sp and s_sp != (None, None)) else 1
            channel_diff = abs((s_ch or 0) - (t_ch or 0))
            # primary weight spatial match, then channel difference
            score = (spatial_mismatch * 1000) + channel_diff
            if best_score is None or score < best_score:
                best_score = score
                best_t = t_name
        if best_t is not None:
            matched_pairs.append((s_name, best_t))
            used_teacher.add(best_t)
        if len(matched_pairs) >= max_pairs:
            break

    # Prepare final persistent hooks for matched pairs
    student_hooks = []
    teacher_hooks = []
    selected_students = set([s for s, _ in matched_pairs])
    selected_teachers = set([t for _, t in matched_pairs])

    # move corresponding FeatureHook objects to final lists and remove other hooks
    for name, hook in list(student_hooks_all.items()):
        if name in selected_students:
            student_hooks.append(hook)    # keep this hook
        else:
            hook.close()                  # remove temporary hook

    for name, hook in list(teacher_hooks_all.items()):
        if name in selected_teachers:
            teacher_hooks.append(hook)
        else:
            hook.close()

    # return in same order (student_hooks[i] corresponds to teacher_hooks[i])
    # reorder teacher_hooks to match matched_pairs order
    name_to_teacher_hook = {h.module._get_name() + str(id(h.module)): h for h in teacher_hooks}  # fallback mapping
    # create mapping by name (we have teacher_hooks_all originally, so we can use matched_pairs directly)
    teacher_hooks_ordered = []
    student_hooks_ordered = []
    teacher_hooks_map = {name: hook for name, hook in teacher_hooks_all.items() if name in selected_teachers}
    student_hooks_map = {name: hook for name, hook in student_hooks_all.items() if name in selected_students}

    for s_name, t_name in matched_pairs:
        student_hooks_ordered.append(student_hooks_map[s_name])
        teacher_hooks_ordered.append(teacher_hooks_map[t_name])

    return student_hooks_ordered, teacher_hooks_ordered, matched_pairs

# -----------------------------
# Simple factory
# -----------------------------
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

    # Minimal usage example for feature distillation (small smoke test)
    try:
        # pick some candidate layers for MobileNetV2 (these indices may vary by torchvision version)
        student_hooks = register_hooks_by_name(student, ["features.4", "features.7"])
        teacher_hooks = []
        # attempt to pick analogous layers on teacher; needs manual adjustment per teacher architecture
        # fallback: register the last convolutional block if available
        if hasattr(teacher, "features"):
            teacher_hooks = register_hooks_by_name(teacher, ["features.6", "features.8"])  # may not exist for Xception
        else:
            # try last conv layer heuristic for timm models
            for name, module in reversed(list(teacher.named_modules())):
                if isinstance(module, nn.Conv2d):
                    teacher_hooks = [FeatureHook(module)]
                    break

        # run a forward pass to populate hooks
        dummy = torch.randn(2, 3, 160, 160).to(device)
        student = student.to(device); teacher = teacher.to(device)
        with torch.no_grad():
            _ = teacher(dummy)
        _ = student(dummy)  # student forward will trigger its hooks

        # gather features
        s_feats = [h.features for h in student_hooks]
        t_feats = [h.features for h in teacher_hooks[:len(s_feats)]]

        # compute feature loss (if both lists non-empty)
        if s_feats and t_feats:
            f_loss = feature_distillation_loss(s_feats, t_feats)
            print("Feature distillation loss (example):", float(f_loss.item()))
        else:
            print("Could not collect matching feature hooks for demo (adjust module names).")

        # cleanup hooks
        for h in student_hooks: h.close()
        for h in teacher_hooks: h.close()
    except Exception as e:
        print("Feature-distill example skipped due to:", e)
