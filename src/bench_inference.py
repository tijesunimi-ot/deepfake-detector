# src/bench_inference.py
import argparse, time
import torch
import numpy as np
from src.models import get_model, get_device

def run(model, device, batch=32, iters=100, warmup=20, size=160):
    x = torch.randn(batch,3,size,size, device=device)
    # warmup
    for _ in range(warmup):
        y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    # measure
    t0 = time.time()
    for _ in range(iters):
        y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.time()
    sec = t1 - t0
    ips = batch * iters / sec
    lat_ms = 1000.0 * sec / iters
    return ips, lat_ms

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=False)
    ap.add_argument("--model", default="mobilenet_v2")
    ap.add_argument("--mode", choices=["fp32_cpu","fp32_cuda","int8_ts"], default="fp32_cpu")
    ap.add_argument("--ts_path", default=None, help="TorchScript for int8_ts")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--size", type=int, default=160)
    args = ap.parse_args()

    if args.mode == "int8_ts":
        assert args.ts_path, "Provide --ts_path for TorchScript INT8"
        model = torch.jit.load(args.ts_path, map_location="cpu")
        model.eval()
        ips, lat = run(model, torch.device("cpu"), batch=args.batch, iters=args.iters, size=args.size)
        print(f"[INT8 CPU] {ips:.1f} imgs/s  ~{lat:.2f} ms/batch")
    else:
        device = torch.device("cuda") if (args.mode=="fp32_cuda" and torch.cuda.is_available()) else torch.device("cpu")
        model = get_model(args.model, num_classes=2, pretrained=False)
        if args.weights:
            sd = torch.load(args.weights, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }
            model.load_state_dict(sd, strict=False)
        model.eval().to(device)
        ips, lat = run(model, device, batch=args.batch, iters=args.iters, size=args.size)
        print(f"[{args.mode.upper()}] {ips:.1f} imgs/s  ~{lat:.2f} ms/batch")

# command to run: python src/bench_inference.py --mode fp32_cuda --weights checkpoints/mobilenet_distill/model_best.pth.tar
