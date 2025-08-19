# src/infer_onnx.py
import onnxruntime as ort
import numpy as np
import cv2
import argparse
from pathlib import Path

# ImageNet normalization (albumentations.Normalize default)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_img_bgr(img_bgr, size=160):
    # img_bgr: OpenCV BGR uint8
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW
    img = img.transpose(2,0,1).astype(np.float32)
    return img

def run_onnx(onnx_path, images, provider_preference=None):
    # provider_preference example: ["CUDAExecutionProvider","CPUExecutionProvider"] if using GPU
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=provider_preference or ort.get_available_providers())
    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    # run
    outputs = sess.run([out_name], {input_name: images})
    return outputs[0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--use_cuda", action="store_true", help="Prefer CUDAExecutionProvider if available")
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit("Failed to read image: " + args.img)
    inp = preprocess_img_bgr(img, size=args.size)
    inp = np.expand_dims(inp, axis=0).astype(np.float32)  # (1,3,H,W)

    providers = None
    if args.use_cuda:
        # try CUDA then CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_out = run_onnx(args.onnx, inp, provider_preference=providers)
    # sess_out shape: (1, num_classes)
    logits = np.array(sess_out).squeeze()
    # softmax to get prob(fake)
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    print("logits:", logits, "prob(fake):", float(probs[1]))

# command to run: python src/infer_onnx.py --onnx checkpoints/onnx/mobilenet_v2.onnx --img test.jpg --use_cuda
# optional: install onnxruntime-gpu for CUDA support
# pip install onnxruntime-gpu