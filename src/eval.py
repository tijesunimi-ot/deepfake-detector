# src/eval.py
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from src.dataset import DeepfakeDataset
from src.models import get_model, get_device

# -------------------------
# Helpers
# -------------------------
def load_checkpoint_weights(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # we accept either bare state_dict or wrapped checkpoint (train.py style)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    else:
        return ckpt  # assume it's a state_dict

def ensure_state_dict_keys(state_dict):
    # remove any "module." prefix from DataParallel keys
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new_state[nk] = v
    return new_state

def predict_batch(model, imgs, device):
    model.eval()
    with torch.no_grad():
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # prob(fake)
        preds = (probs >= 0.5).astype(int)
    return probs, preds

def plot_roc(y_true, y_scores, out_path, title="ROC"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else float('nan')
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return auc

# -------------------------
# Main evaluation
# -------------------------
def run_evaluation(weights, model_name, test_csv, cfg_yaml=None, batch_size=32, num_workers=4, 
                   device=None, aggregate_method="mean", out_dir="results/eval"):
    device = device if device is not None else str(get_device())
    device = torch.device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load config if provided (to get num_classes / image size etc)
    cfg = {}
    if cfg_yaml:
        with open(cfg_yaml, 'r') as f:
            cfg = yaml.safe_load(f)

    test_csv = Path(test_csv)
    assert test_csv.exists(), f"{test_csv} does not exist"

    # deterministic transforms for eval
    size = cfg.get("data", {}).get("crop_size", 160)
    # If crop size not in cfg, default to 160; DeepfakeDataset will still resize if necessary
    eval_transforms = Compose([
        Resize(size, size),
        Normalize(),  # default ImageNet mean/std in albumentations
        ToTensorV2()
    ])

    test_ds = DeepfakeDataset(str(test_csv), transforms=eval_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Load model
    print("Instantiating model:", model_name)
    model = get_model(model_name, num_classes=cfg.get("model", {}).get("num_classes", 2), pretrained=False)
    # load weights
    state = load_checkpoint_weights(weights)
    state = ensure_state_dict_keys(state)
    try:
        model.load_state_dict(state)
    except Exception as e:
        # try partial loading (in case the checkpoint contained extra keys)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and v.size() == model_state[k].size()}
        model_state.update(filtered)
        model.load_state_dict(model_state)
        print("Warning: partial state_dict loaded (filtered mismatched keys).")
    model = model.to(device)
    model.eval()

    # iterate and collect predictions
    rows = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for imgs, labels in pbar:
            probs, preds = predict_batch(model, imgs, device)
            # imgs come from dataset that preserves order with test_csv rows, so we can iterate in sync
            # but DeepfakeDataset doesn't expose current paths; so we will re-read test_csv directly to map indexes
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    # build predictions dataframe from test_csv (which has same order as dataset)
    df_meta = pd.read_csv(test_csv)
    assert len(df_meta) == len(all_probs), "Mismatch: metadata rows and predictions length differ"
    df_meta = df_meta.copy()
    df_meta["prob_fake"] = all_probs
    df_meta["pred_label"] = (df_meta["prob_fake"] >= 0.5).astype(int)

    # save per-frame predictions
    preds_csv = out_dir / "preds_frame_level.csv"
    df_meta.to_csv(preds_csv, index=False)
    print("Saved frame-level predictions to", preds_csv)

    # Frame-level metrics
    y_true = df_meta["label"].values
    y_scores = df_meta["prob_fake"].values
    frame_auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else float('nan')
    frame_preds = (y_scores >= 0.5).astype(int)
    frame_acc = accuracy_score(y_true, frame_preds)
    cm = confusion_matrix(y_true, frame_preds)
    print(f"Frame-level: AUC={frame_auc:.4f}, ACC={frame_acc:.4f}")
    # save confusion matrix
    pd.DataFrame(cm, index=["real","fake"], columns=["pred_real","pred_fake"]).to_csv(out_dir / "confusion_matrix_frame.csv")

    # Plot overall ROC
    overall_roc = out_dir / "roc_overall.png"
    try:
        auc_val = plot_roc(y_true, y_scores, overall_roc, title="Overall ROC (frame-level)")
    except Exception as e:
        print("ROC plot failed:", e)
        auc_val = frame_auc

    # Per-manipulation metrics
    per_manip = []
    for manip, g in df_meta.groupby("manipulation"):
        y_t = g["label"].values
        y_s = g["prob_fake"].values
        if len(np.unique(y_t)) <= 1:
            auc_m = float('nan')
        else:
            auc_m = roc_auc_score(y_t, y_s)
        acc_m = accuracy_score(y_t, (y_s >= 0.5).astype(int))
        per_manip.append({"manipulation": manip, "count": len(g), "auc": auc_m, "acc": acc_m})
        # plot per-manip ROC
        try:
            plot_roc(y_t, y_s, out_dir / f"roc_{manip}.png", title=f"ROC ({manip})")
        except Exception:
            pass
    df_per_manip = pd.DataFrame(per_manip).sort_values("count", ascending=False)
    df_per_manip.to_csv(out_dir / "per_manipulation_metrics.csv", index=False)
    print("Saved per-manipulation metrics to", out_dir / "per_manipulation_metrics.csv")

    # Video-level aggregation
    if "video" in df_meta.columns:
        agg_rows = []
        grouped = df_meta.groupby("video")
        for v, g in grouped:
            if aggregate_method == "mean":
                agg_score = g["prob_fake"].mean()
            elif aggregate_method == "median":
                agg_score = g["prob_fake"].median()
            elif aggregate_method == "max":
                agg_score = g["prob_fake"].max()
            else:
                agg_score = g["prob_fake"].mean()
            true_label = int(g["label"].mode().iloc[0])  # assume majority label per video
            agg_rows.append({"video": v, "manipulation": g["manipulation"].iloc[0], "prob_fake": agg_score, "label": true_label})

        df_video = pd.DataFrame(agg_rows)
        df_video["pred_label"] = (df_video["prob_fake"] >= 0.5).astype(int)
        video_auc = roc_auc_score(df_video["label"].values, df_video["prob_fake"].values) if len(np.unique(df_video["label"].values)) > 1 else float('nan')
        video_acc = accuracy_score(df_video["label"].values, df_video["pred_label"].values)
        df_video.to_csv(out_dir / "preds_video_level.csv", index=False)
        pd.DataFrame(confusion_matrix(df_video["label"].values, df_video["pred_label"].values), index=["real","fake"], columns=["pred_real","pred_fake"]).to_csv(out_dir / "confusion_matrix_video.csv")
        print(f"Video-level ({aggregate_method}) AUC={video_auc:.4f}, ACC={video_acc:.4f}")
        # plot video-level ROC
        try:
            plot_roc(df_video["label"].values, df_video["prob_fake"].values, out_dir / "roc_video_level.png", title="Video-level ROC")
        except Exception:
            pass
    else:
        video_auc = float('nan')
        video_acc = float('nan')

    # Save summary
    summary = {
        "frame_level": {"auc": float(frame_auc) if not np.isnan(frame_auc) else None, "acc": float(frame_acc)},
        "video_level": {"auc": float(video_auc) if not np.isnan(video_auc) else None, "acc": float(video_acc)},
        "n_frames": len(df_meta),
        "n_videos": int(df_meta["video"].nunique()) if "video" in df_meta.columns else None
    }
    with open(Path(out_dir) / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved evaluation summary to", out_dir / "eval_summary.json")
    print("Evaluation complete.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth or .pth.tar)")
    parser.add_argument("--model", default="mobilenet_v2", help="model name used in src.models.get_model")
    parser.add_argument("--test_csv", default="data/processed/test_meta.csv", help="test metadata CSV")
    parser.add_argument("--config", default=None, help="optional config YAML (to adapt size/num_classes)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--aggregate", default="mean", choices=["mean","median","max"])
    parser.add_argument("--out_dir", default="results/eval")
    args = parser.parse_args()
    run_evaluation(args.weights, args.model, args.test_csv, cfg_yaml=args.config,
                   batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
                   aggregate_method=args.aggregate, out_dir=args.out_dir)
