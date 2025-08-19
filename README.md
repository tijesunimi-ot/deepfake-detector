# Deepfake Detector — README

A compact summary tying together the full end-to-end PyTorch system (data → train → eval → compression → quantization → export → demo + Docker).

---

## Project overview

Lightweight deepfake detection pipeline built around a **MobileNetV2 student** (optionally distilled from an **Xception / ResNet teacher**). Targets real-time/near-real-time deployment with pruning, quantization, ONNX export and GPU/TensorRT options.

Primary goals:
- Small model footprint & low latency
- Reproducible training + evaluation
- Practical export paths (ONNX, TorchScript, TensorRT)
- Dockerized runtime for GPU inference

---

## Quick repo map (key files)

```text
deepfake-detector/
├── data/raw/ffpp/                      # your FF++ download (original, deepfakes, faceswap, ...)
├── data/processed/                     # frames, crops, metadata.csv, train/val/test metas
├── configs/                            # yaml configs (mobilenet_distill.yaml, etc.)
├── src/
│   ├── preprocess.py                   # Step 2: frame extract + MTCNN crops -> metadata.csv
│   ├── split_dataset.py                # video-level train/val/test split
│   ├── dataset.py                      # PyTorch Dataset (albumentations)
│   ├── models.py                       # MobileNetV2 student, Xception teacher, DistillationLoss
│   ├── train.py                        # training + optional distillation (mixed-precision)
│   ├── eval.py                         # Step 5: frame/video/per-manipulation evaluation + ROC plots
│   ├── prune.py                        # Step 6: one-shot + IMP pruning scaffold
│   ├── distill_helpers.py              # feature-distillation hooks (optional)
│   ├── quant_utils.py                  # quantization helpers (fuse)
│   ├── quantize_ptq.py                 # Step 7: post-training quantization (INT8 CPU)
│   ├── quantize_qat.py                 # Step 7: quantization-aware training (QAT)
│   ├── export.py                       # Step 8: export to ONNX
│   ├── infer_onnx.py                   # ONNX Runtime single image inference
│   ├── webcam_demo.py                  # real-time demo (MTCNN -> ONNX)
│   └── bench_inference.py              # micro-benchmark tool
├── checkpoints/                         # saved models, pruned, quantized, onnx/
├── notebooks/                           # optional Colab experiments
├── docker/                              # Dockerfiles + entrypoint + docker-compose
└── README.md                            # this file
```

---

## Quickstart (local dev)

1. Create & activate python env:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# install torch matching your CUDA per https://pytorch.org/get-started/locally
```

2. Confirm GPU (optional):

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## Step 1 — Preprocess FF++ (you already have the dataset)

Assumes your FF++ folder is `data/raw/ffpp/` with subfolders `original`, `deepfakes`, `faceswap`, etc.

```bash
# extract frames + face crops -> data/processed/metadata.csv
python src/preprocess.py --raw_root data/raw/ffpp --out_root data/processed --frames_step 3 --crop_size 160

# then split by video (train/val/test)
python src/split_dataset.py --meta_csv data/processed/metadata.csv --out_root data/processed --val_size 0.1 --test_size 0.1
```

Notes:
- `frames_step=3` keeps every 3rd frame to save disk.
- Crops are saved under `data/processed/crops/<manipulation>/<video>/`.

---

## Step 2 — Train (distillation or plain CE)

Edit `configs/mobilenet_distill.yaml` for hyperparameters (batch size, epochs, teacher path, etc.).

Run:

```bash
python src/train.py --config configs/mobilenet_distill.yaml
```

Recommendations for GTX 1650 (4GB VRAM):
- `batch_size`: 8–16 (reduce if OOM)
- `amp: true` (mixed precision)
- Start with `freeze_backbone: true` for quick tests, then unfreeze

To resume:
```bash
# set cfg.resume to checkpoint path or pass resume via code modifications
```

---

## Step 3 — Evaluation

Evaluate a trained checkpoint (frame + video level + per-manipulation):

```bash
python src/eval.py \
  --weights checkpoints/mobilenet_distill/model_best.pth.tar \
  --model mobilenet_v2 \
  --test_csv data/processed/test_meta.csv \
  --config configs/mobilenet_distill.yaml \
  --out_dir results/eval
```

Outputs:
- `results/eval/preds_frame_level.csv`
- `results/eval/preds_video_level.csv`
- ROC plots & `eval_summary.json` + per-manipulation metrics

---

## Step 4 — Compression (pruning + distillation)

Workflow:
1. Train a strong teacher (Xception/ResNet) with `train.py` (set `use_teacher: false`).
2. Distill teacher → student using `mobilenet_distill.yaml` (`use_teacher: true`, `teacher_weights: <teacher.ckpt>`).
3. Prune the distilled student:

One-shot prune:
```bash
python src/prune.py --mode one_shot --weights_in checkpoints/mobilenet_distill/model_best.pth.tar --weights_out checkpoints/pruned/pruned_60.pth --amount 0.6
```

Iterative Magnitude Pruning (IMP):
- Implement `finetune_fn` (or call `train.py` from inside `prune.py`) and run IMP (script scaffolds provided).

Notes:
- Unstructured pruning reduces storage but may not reduce latency unless using sparse-aware runtimes.
- For real runtime speedups, use structured channel pruning + model surgery (I can provide a script on request).

---

## Step 5 — Quantization (PTQ / QAT)

PTQ (fast):

```bash
python src/quantize_ptq.py --weights checkpoints/mobilenet_distill/model_best.pth.tar --calib_csv data/processed/val_meta.csv --test_csv data/processed/test_meta.csv --size 160 --calib_batches 100
```

QAT (if PTQ loses too much accuracy):

```bash
python src/quantize_qat.py --weights checkpoints/mobilenet_distill/model_best.pth.tar --epochs 5 --batch_size 16 --cuda --amp --out_dir checkpoints/quantized_qat
```

Outputs: TorchScript INT8 models (`checkpoints/quantized/*_int8_*.pt`) for CPU inference.

---

## Step 6 — Export & runtime (ONNX / TensorRT / demo)

Export ONNX:

```bash
python src/export.py --weights checkpoints/mobilenet_distill/model_best.pth.tar --out checkpoints/onnx/mobilenet_v2.onnx --size 160 --dynamic
```

Test ONNX runtime (single image):

```bash
python src/infer_onnx.py --onnx checkpoints/onnx/mobilenet_v2.onnx --img test.jpg --use_cuda
```

Webcam demo (ONNX runtime + MTCNN):

```bash
python src/webcam_demo.py --onnx checkpoints/onnx/mobilenet_v2.onnx --use_cuda
```

TensorRT (optional, best GPU perf):

```bash
trtexec --onnx=checkpoints/onnx/mobilenet_v2.onnx --saveEngine=checkpoints/trt/mobilenet_v2_fp16.trt --fp16 --workspace=4096 --explicitBatch
```

---

## Step 7 — Docker (runtime)

Two recommended options are included:
- **ONNX Runtime GPU** Dockerfile (`docker/Dockerfile.onnxruntime_gpu`) — easiest to run ONNX GPU workloads.
- **NGC TensorRT** container — for maximum TensorRT performance (pull from NGC).

Example run (webcam demo):

```bash
# build
docker build -f docker/Dockerfile.onnxruntime_gpu -t deepfake:onnxruntime .

# run (exposes webcam & display)
xhost +local:docker
docker run --gpus all --rm -it \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  deepfake:onnxruntime \
  demo /workspace/checkpoints/onnx/mobilenet_v2.onnx
```

(See `docker/entrypoint.sh` for details.)

---

## Troubleshooting / tips
- If `torch.cuda.is_available()` is False: verify drivers and CUDA toolkit on host and install matching PyTorch wheel.
- If OOM: reduce `batch_size`, reduce `size` to 128, or use mixed precision (`amp`).
- If ONNX export / TensorRT fails: try `opset=11` or run `onnx-simplifier`.
- If eval AUC ≈ 0.5: verify labels, check for data leakage (same video in train+test), ensure preprocessing matches training (mean/std & resize).
- For real speedups on GPU: convert ONNX → TensorRT FP16 (trtexec) and run the engine in Python or with TensorRT sample code.

---

## Ethics & usage
This code is intended for research and defensive use (detecting manipulated media). Before deploying:
- Respect dataset licenses (FaceForensics++ / DFDC).
- Consider user privacy, consent, and legal implications.
- Avoid any application that harms individuals or enables misuse.

---

## Next steps & contact
If you want I can:
- add a scripted MobileNetV2 model surgery for structured-pruned speedups,  
- produce a Colab notebook that runs the full pipeline on a small sample, or  
- create a production-ready TensorRT Python inference script and example server (FastAPI + TensorRT).

Say which and I’ll produce that next (one step at a time).

