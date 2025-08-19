# src/train.py
import os
import time
import argparse
import yaml
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# project modules
from src.dataset import DeepfakeDataset
from src.models import get_model, DistillationLoss, get_device

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# ---------------------------
# Helpers
# ---------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, is_best, out_dir, fname="checkpoint.pth.tar"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    torch.save(state, str(path))
    if is_best:
        best_path = out_dir / "model_best.pth.tar"
        torch.save(state, str(best_path))

def load_checkpoint(path, model=None, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    if model is not None and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint

def compute_metrics(y_true, y_pred_logits):
    """
    y_pred_logits: Nx2 logits for classes [real, fake]
    returns dict with acc and auc
    """
    probs = torch.softmax(torch.from_numpy(y_pred_logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")
    return {"acc": acc, "auc": auc, "probs": probs}

# ---------------------------
# Train / Validate loops
# ---------------------------
def train_one_epoch(epoch, model, teacher, loader, optimizer, criterion, device, scaler, cfg, writer=None):
    model.train()
    if teacher is not None:
        teacher.eval()
    losses = []
    all_logits = []
    all_labels = []
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Epoch {epoch}")
    for i, (imgs, labels) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with amp.autocast(enabled=cfg['amp']):
            student_logits = model(imgs)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(imgs)
                loss, loss_ce, loss_kl = criterion(student_logits, teacher_logits, labels)
            else:
                # standard CE
                loss = nn.CrossEntropyLoss()(student_logits, labels)
                loss_ce = loss
                loss_kl = torch.tensor(0.0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))
        all_logits.append(student_logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        if writer is not None:
            global_step = epoch * len(loader) + i
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/loss_ce", float(loss_ce.item()) if hasattr(loss_ce, "item") else float(loss_ce), global_step)
            if isinstance(loss_kl, torch.Tensor):
                writer.add_scalar("train/loss_kl", float(loss_kl.item()), global_step)

        pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_labels, all_logits)
    metrics['loss'] = float(np.mean(losses))
    return metrics

def validate(model, loader, device, cfg, return_preds=False):
    model.eval()
    losses = []
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validate"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = nn.CrossEntropyLoss()(logits, labels)
            losses.append(float(loss.item()))
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_labels, all_logits)
    metrics['loss'] = float(np.mean(losses))
    if return_preds:
        return metrics, all_logits, all_labels
    return metrics

# ---------------------------
# Main
# ---------------------------
def main(cfg):
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print("Using device:", device)

    # Datasets & loaders
    train_ds = DeepfakeDataset(cfg['data']['train_meta'], transforms=None)  # dataset uses default transforms if None
    val_ds = DeepfakeDataset(cfg['data']['val_meta'], transforms=None)
    # Recommended safe batch size for GTX1650: 8 or 16 for 160x160; reduce if OOM
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True,
                              num_workers=cfg['data'].get('num_workers', 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False,
                            num_workers=cfg['data'].get('num_workers', 4), pin_memory=True)

    # Models
    student = get_model(cfg['model']['student'], num_classes=cfg['model'].get('num_classes', 2),
                        pretrained=cfg['model'].get('pretrained_backbone', True),
                        freeze_backbone=cfg['model'].get('freeze_backbone', False))
    student = student.to(device)

    teacher = None
    criterion = None
    if cfg['training'].get('use_teacher', False):
        teacher = get_model(cfg['model']['teacher'], num_classes=cfg['model'].get('num_classes', 2),
                            pretrained=cfg['model'].get('teacher_pretrained', True))
        teacher = teacher.to(device)
        if cfg['training'].get('teacher_weights'):
            print("Loading teacher weights from", cfg['training']['teacher_weights'])
            map_loc = None if torch.cuda.is_available() else 'cpu'
            load_checkpoint(cfg['training']['teacher_weights'], model=teacher, map_location=map_loc)
        teacher.eval()
        criterion = DistillationLoss(temperature=cfg['training'].get('T', 4.0),
                                     alpha=cfg['training'].get('alpha', 0.6))
    else:
        criterion = None

    # Optimizer & scheduler
    optim_cfg = cfg['training'].get('optimizer', {})
    opt_name = optim_cfg.get('name', 'adam').lower()
    lr = optim_cfg.get('lr', 1e-3)
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=optim_cfg.get('momentum', 0.9), weight_decay=optim_cfg.get('weight_decay', 1e-4))
    else:
        optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=optim_cfg.get('weight_decay', 1e-6))

    sched_cfg = cfg['training'].get('scheduler', {})
    sched_name = sched_cfg.get('name', 'cosine')
    if sched_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training'].get('epochs', 20))
    else:
        scheduler = None

    # Mixed precision
    scaler = amp.GradScaler(enabled=cfg['amp'])

    # Logging
    writer = None
    if cfg.get('log_dir') and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=cfg['log_dir'])

    best_auc = -1.0
    out_dir = cfg.get('out_dir', 'checkpoints')

    # optionally resume
    start_epoch = 0
    if cfg.get('resume'):
        ckpt = load_checkpoint(cfg['resume'], model=student, optimizer=optimizer, map_location=torch.device(device).type)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc = ckpt.get("best_auc", best_auc)
        print(f"Resumed from {cfg['resume']}, starting at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, cfg['training'].get('epochs', 20)):
        t0 = time.time()
        train_metrics = train_one_epoch(epoch, student, teacher, train_loader, optimizer,
                                        criterion if criterion is not None else nn.CrossEntropyLoss(),
                                        device, scaler, cfg, writer=writer)
        val_metrics, val_labels, val_logits = validate(student, val_loader, device, cfg, return_preds=True)

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        # Logging
        print(f"Epoch {epoch} train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} auc {train_metrics['auc']:.4f}")
        print(f"Epoch {epoch} val   loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} auc {val_metrics['auc']:.4f}")
        if writer is not None:
            # scalars
            writer.add_scalar("val/loss", val_metrics['loss'], epoch)
            writer.add_scalar("val/acc", val_metrics['acc'], epoch)
            writer.add_scalar("val/auc", val_metrics['auc'], epoch)
            writer.add_scalar("train/acc", train_metrics['acc'], epoch)
            writer.add_scalar("train/auc", train_metrics['auc'], epoch)

            #confusion matrix
            preds = (val_metrics['probs'] >= 0.5).astype(int)
            cm = confusion_matrix(val_labels, preds)
            fig, ax = plt.subplots(figsize=(4, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
            disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
            writer.add_figure("val/confusion_matrix", fig, epoch)
            plt.close(fig)

            # Sample predictions (first 16 from last val batch)
            sample_imgs = next(iter(val_loader))[0][:16].to(device)
            with torch.no_grad():
                sample_logits = student(sample_imgs)
                sample_preds = sample_logits.argmax(dim=1)
            
            grid = torchvision.utils.make_grid(sample_imgs.cpu(), nrow=4, normalize=True, scale_each=True)
            writer.add_image("val/sample_images", grid, epoch)
            writer.add_text("val/sample_predictions", str(sample_preds.tolist()), epoch)


        # checkpoint
        is_best = False
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            is_best = True

        ckpt_state = {
            "epoch": epoch,
            "state_dict": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_auc": best_auc,
            "cfg": cfg
        }
        save_checkpoint(ckpt_state, is_best, out_dir, fname=f"checkpoint_epoch{epoch}.pth.tar")

    if writer is not None:
        writer.close()

    print("Training complete. Best val AUC:", best_auc)

# ---------------------------
# CLI / Config loader
# ---------------------------
def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mobilenet_distill.yaml", help="YAML config file")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    main(cfg)
