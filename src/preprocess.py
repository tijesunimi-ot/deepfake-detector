# src/preprocess.py
import os
import argparse
import subprocess
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
from tqdm import tqdm

def extract_frames(video_path, out_dir, step=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg: extract all frames, then we will sample by rename pattern to only keep every step-th
    # but more efficient: use ffmpeg -vf "select=not(mod(n\,step))" -vsync vfr
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"select=not(mod(n\\,{step}))",
        "-vsync", "vfr",
        "-q:v", "2",
        str(out_dir / "frame_%06d.jpg")
    ]
    subprocess.run(cmd, check=True)

def detect_and_save_crops(frames_dir, crops_dir, mtcnn, min_face_size=40, size=160):
    frames = sorted(frames_dir.glob("*.jpg"))
    crops_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fpath in tqdm(frames, desc=f"Detect {frames_dir.name}"):
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            continue
        boxes, probs = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            continue
        # choose the largest box (by area)
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        idx = int(max(range(len(areas)), key=lambda i: areas[i]))
        box = boxes[idx]
        # filter small face
        w = box[2]-box[0]; h = box[3]-box[1]
        if min(w, h) < min_face_size:
            continue
        # square crop with padding
        cx = int((box[0]+box[2])/2); cy = int((box[1]+box[3])/2)
        half = int(max(w, h)/2 * 1.2)
        left = max(cx-half, 0); top = max(cy-half, 0)
        right = min(cx+half, img.width); bottom = min(cy+half, img.height)
        crop = img.crop((left, top, right, bottom)).resize((size, size), Image.BILINEAR)
        out_fname = crops_dir / (fpath.stem + "_crop.jpg")
        crop.save(out_fname, quality=95)
        results.append((str(fpath), str(out_fname)))
    return results

def process_ffpp(raw_root, out_root, frames_step=3, crop_size=160, min_face_size=40):
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    # FF++ layout: raw_root/<manipulation>/<video>.mp4  (adjust if your structure differs)
    for manipulation_dir in sorted(raw_root.iterdir()):
        if not manipulation_dir.is_dir():
            continue
        label = 0 if manipulation_dir.name.lower() == "original" else 1
        for video_path in sorted(manipulation_dir.glob("*.mp4")):
            video_name = video_path.stem
            frames_out = out_root / "frames" / manipulation_dir.name / video_name
            crops_out = out_root / "crops" / manipulation_dir.name / video_name
            # 1) extract frames (sampled)
            try:
                if not any(frames_out.glob("*.jpg")):
                    extract_frames(video_path, frames_out, step=frames_step)
            except subprocess.CalledProcessError as e:
                print(f"ffmpeg failed for {video_path}: {e}")
                continue
            # 2) detect + save crops
            detections = detect_and_save_crops(frames_out, crops_out, mtcnn, min_face_size=min_face_size, size=crop_size)
            # record metadata rows
            for frame_path, crop_path in detections:
                rows.append({
                    "video": video_name,
                    "manipulation": manipulation_dir.name,
                    "frame_path": frame_path,
                    "crop_path": crop_path,
                    "label": label
                })
    df = pd.DataFrame(rows)
    meta_path = out_root / "metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"Saved metadata to {meta_path} with {len(df)} rows")

if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="data/raw/ffpp")
    parser.add_argument("--out_root", type=str, default="data/processed")
    parser.add_argument("--frames_step", type=int, default=3, help="keep every N-th frame")
    parser.add_argument("--crop_size", type=int, default=160)
    parser.add_argument("--min_face_size", type=int, default=40)
    args = parser.parse_args()
    process_ffpp(args.raw_root, args.out_root, frames_step=args.frames_step, crop_size=args.crop_size, min_face_size=args.min_face_size)
