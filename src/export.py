# src/export.py
import torch
from pathlib import Path
import argparse
from src.models import get_model
import onnx
try:
    from onnxsim import simplify
    HAS_ONNXSIM = True
except Exception:
    HAS_ONNXSIM = False

def load_state_to_model(model, weights_path):
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strip possible "module." prefix
    state = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
    model.load_state_dict(state, strict=False)
    return model

def export_onnx(model_name, weights, out_path, input_size=160, opset=13, dynamic_batch=True, simplify_onnx=True):
    model = get_model(model_name, num_classes=2, pretrained=False)
    model = load_state_to_model(model, weights)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        verbose=False
    )
    print("ONNX exported to:", out_path)

    if simplify_onnx and HAS_ONNXSIM:
        print("Running onnx-simplifier...")
        model_onnx = onnx.load(str(out_path))
        model_simp, check = simplify(model_onnx)
        if check:
            onnx.save(model_simp, str(out_path))
            print("ONNX simplified and saved (onxsim).")
        else:
            print("ONNX simplifier check failed; leaving original ONNX.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mobilenet_v2")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="checkpoints/onnx/mobilenet_v2.onnx")
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--dynamic", action="store_true", help="enable dynamic batch dim")
    p.add_argument("--no_simplify", action="store_true", help="skip onnx-simplifier")
    args = p.parse_args()
    export_onnx(args.model, args.weights, args.out, input_size=args.size, opset=args.opset, dynamic_batch=args.dynamic, simplify_onnx=(not args.no_simplify))

# command to run: python src/export.py --weights checkpoints/mobilenet_distill/model_best.pth.tar --out checkpoints/onnx/mobilenet_v2.onnx --size 160 --dynamic
# optional: install onnx-simplifier for smaller, cleaner ONNX:
# pip install onnx-simplifier