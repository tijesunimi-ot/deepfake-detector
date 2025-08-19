# src/quantize_ptq.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from src.models import get_model, get_device
from src.dataset import DeepfakeDataset
from src.quant_utils import fuse_mobilenetv2

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob_fake = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(prob_fake.tolist())
    ys = np.array(ys); ps = np.array(ps)
    acc = accuracy_score(ys, (ps>=0.5).astype(int))
    auc = roc_auc_score(ys, ps) if len(np.unique(ys))>1 else float('nan')
    return acc, auc

def main(args):
    cpu = torch.device("cpu")
    # deterministic eval transforms
    size = args.size
    tfm = Compose([Resize(size, size), Normalize(), ToTensorV2()])

    # small calibration & test loaders
    calib_ds = DeepfakeDataset(args.calib_csv, transforms=tfm)
    test_ds  = DeepfakeDataset(args.test_csv,  transforms=tfm)

    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Load FP32 model and weights (on CPU for quant)
    model_fp32 = get_model(args.model, num_classes=2, pretrained=False)
    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strip "module." if present
    state = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
    model_fp32.load_state_dict(state, strict=False)
    model_fp32.eval()

    # Baseline FP32 (CPU) eval
    acc32, auc32 = evaluate(model_fp32, test_loader, cpu)
    print(f"[FP32 CPU] ACC={acc32:.4f}  AUC={auc32:.4f}")

    # Fuse modules
    fuse_mobilenetv2(model_fp32)

    # Static quantization
    torch.backends.quantized.engine = "fbgemm"  # x86
    qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    model_fp32.qconfig = qconfig
    prepared = torch.ao.quantization.prepare(model_fp32, inplace=False)

    # Calibration (no grads)
    with torch.no_grad():
        n_batches = 0
        for x, _ in calib_loader:
            prepared(x)  # run forward to collect stats
            n_batches += 1
            if n_batches >= args.calib_batches:
                break

    # Convert to INT8
    quantized = torch.ao.quantization.convert(prepared, inplace=False)

    # Evaluate INT8 (CPU only)
    acc8, auc8 = evaluate(quantized, test_loader, cpu)
    print(f"[INT8 PTQ] ACC={acc8:.4f}  AUC={auc8:.4f}  (calib_batches={args.calib_batches})")

    # Save TorchScript for deployment
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1,3,size,size)
    ts = torch.jit.trace(quantized, example)
    ts_path = out_dir / f"{args.model}_int8_ptq_ts.pt"
    ts.save(str(ts_path))
    sd_path = out_dir / f"{args.model}_int8_ptq_state.pth"
    torch.save(quantized.state_dict(), sd_path)
    print("Saved INT8 PTQ TorchScript:", ts_path)
    print("Saved INT8 PTQ state_dict:", sd_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="mobilenet_v2")
    ap.add_argument("--calib_csv", default="data/processed/val_meta.csv")
    ap.add_argument("--test_csv",  default="data/processed/test_meta.csv")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--calib_batches", type=int, default=50)
    ap.add_argument("--out_dir", default="checkpoints/quantized")
    args = ap.parse_args()
    main(args)

#command to run scrpt: python src/quantize_ptq.py \
# --weights checkpoints/mobilenet_distill/model_best.pth.tar \
# --calib_csv data/processed/val_meta.csv \
# --test_csv  data/processed/test_meta.csv \
# --size 160 --batch_size 64 --calib_batches 100 \
# --out_dir checkpoints/quantized
