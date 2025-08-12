# Deepfake Detector (PyTorch)

Step-by-step project to train and deploy a lightweight MobileNet-based deepfake detector.
See docs/ for instructions.

Quickstart:
1. Create & activate venv
2. pip install -r requirements.txt
3. Fill data/ with samples or run scripts/download_data.sh

Run training:
python src/train.py --config configs/mobilenet_train.yaml

Run inference:
python src/infer.py --weights checkpoints/mobilenet_best.pth --image test.jpg