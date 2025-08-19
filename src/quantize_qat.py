# src/quantize_qat.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from src.models import get_model
from src.dataset import DeepfakeDataset
from src.quant_utils import fuse_mobilenetv2

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob_fake = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(prob_fake.tolist())
    ys = np.array(ys); ps = np.array(ps)
    acc = accuracy_score(ys, (ps>=0.5).astype(int))
    auc = roc_auc_score(ys, ps) if len(np.unique(ys))>1 else float('nan')
    return acc, auc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    size = args.size
    tfm_train = Compose([Resize(size, size), Normalize(), ToTensorV2()])
    tfm_eval  = Compose([Resize(size, size), Normalize(), ToTensorV2()])

    train_ds = DeepfakeDataset(args.train_csv, transforms=tfm_train)
    val_ds   = DeepfakeDataset(args.val_csv,   transforms=tfm_eval)
    test_ds  = DeepfakeDataset(args.test_csv,  transforms=tfm_eval)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Load FP32 model & weights
    model = get_model(args.model, num_classes=2, pretrained=False)
    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
    model.load_state_dict(state, strict=False)

    # Fuse & QAT prepare
    fuse_mobilenetv2(model)
    torch.backends.quantized.engine = "fbgemm"
    qat_qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    model.qconfig = qat_qconfig
    model = torch.ao.quantization.prepare_qat(model, inplace=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler(enabled=args.amp)

    ce = nn.CrossEntropyLoss()

    # Short QAT fine-tuning
    best_auc = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        acc, auc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] train_loss={np.mean(losses):.4f}  val_acc={acc:.4f}  val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), Path(args.out_dir) / "qat_pre_convert_best_state.pth")

    # Convert to int8
    model.cpu()  # conversion on CPU
    quantized = torch.ao.quantization.convert(model.eval(), inplace=False)

    # Final eval on CPU INT8
    acc8, auc8 = evaluate(quantized, test_loader, torch.device("cpu"))
    print(f"[INT8 QAT] test_acc={acc8:.4f}  test_auc={auc8:.4f}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ex = torch.randn(1,3,size,size)
    ts = torch.jit.trace(quantized, ex)
    ts_path = out_dir / f"{args.model}_int8_qat_ts.pt"
    ts.save(str(ts_path))
    sd_path = out_dir / f"{args.model}_int8_qat_state.pth"
    torch.save(quantized.state_dict(), sd_path)
    print("Saved INT8 QAT TorchScript:", ts_path)
    print("Saved INT8 QAT state_dict:", sd_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="mobilenet_v2")
    ap.add_argument("--train_csv", default="data/processed/train_meta.csv")
    ap.add_argument("--val_csv",   default="data/processed/val_meta.csv")
    ap.add_argument("--test_csv",  default="data/processed/test_meta.csv")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)           # short QAT tune
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--out_dir", default="checkpoints/quantized_qat")
    args = ap.parse_args()
    main(args)

#script to run (few epochs): python src/quantize_qat.py \
# --weights checkpoints/mobilenet_distill/model_best.pth.tar \
# --epochs 5 --batch_size 16 --lr 1e-4 --amp --cuda \
# --out_dir checkpoints/quantized_qat
