# src/preprocess.py
import os
import argparse
import subprocess
import shutil
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import torch
import pandas as pd

def extract_frames(video_path, out_dir, step=3):
    out_dir.mkdir(parents=True, exist_ok=True)
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
    saved = 0
    for fpath in tqdm(frames, desc=f"Detect {frames_dir.name}", leave=False):
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
        # Save (overwrite if exists)
        crop.save(out_fname, quality=95)
        saved += 1
    return saved

def count_jpgs(dir_path):
    if not dir_path.exists():
        return 0
    return sum(1 for _ in dir_path.glob("*.jpg"))

def safe_move_dir(src: Path, dst: Path):
    if dst.exists():
        raise FileExistsError(f"Destination {dst} already exists.")
    shutil.move(str(src), str(dst))

def build_metadata_rows_for_video(frames_out: Path, crops_out: Path, video_name: str, manipulation: str, label: int):
    """Return a list of dict rows for this video to be appended to CSV"""
    rows = []
    for crop_fp in sorted(crops_out.glob("*.jpg")):
        crop_name = crop_fp.name  # e.g., frame_000123_crop.jpg
        if crop_name.endswith("_crop.jpg"):
            frame_name = crop_name.replace("_crop.jpg", ".jpg")
            frame_path = frames_out / frame_name
        else:
            frame_path = frames_out / crop_name
        rows.append({
            "video": video_name,
            "manipulation": manipulation,
            "frame_path": str(frame_path),
            "crop_path": str(crop_fp),
            "label": label
        })
    return rows

def append_metadata_rows(out_root: Path, rows):
    """Append rows (list of dict) to data/processed/metadata.csv atomically (append mode)."""
    if not rows:
        return
    meta_path = out_root / "metadata.csv"
    df = pd.DataFrame(rows)
    df.to_csv(meta_path, mode='a', header=not meta_path.exists(), index=False)

def try_delete_frames(frames_out: Path):
    """Delete frames directory if it exists (used when cleanup flag is enabled)."""
    if frames_out.exists():
        try:
            shutil.rmtree(frames_out)
            tqdm.write(f"Deleted frames directory: {frames_out}")
        except Exception as e:
            tqdm.write(f"Warning: failed to delete frames {frames_out}: {e}")

def process_ffpp(raw_root, out_root, frames_step=3, crop_size=160, min_face_size=40, min_crops=5, cleanup_frames=False):
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    log_path = out_root / "processed_videos.txt"
    processed_videos = set()
    if log_path.exists():
        with open(log_path, "r") as f:
            processed_videos = set(line.strip() for line in f if line.strip())

    # FF++ layout: raw_root/<manipulation>/<video>.mp4
    for manipulation_dir in sorted(raw_root.iterdir()):
        if not manipulation_dir.is_dir():
            continue
        label = 0 if manipulation_dir.name.lower() == "original" else 1
        for video_path in sorted(manipulation_dir.glob("*.mp4")):
            video_name = video_path.stem
            rel_video_name = f"{manipulation_dir.name}/{video_name}.mp4"
            frames_out = out_root / "frames" / manipulation_dir.name / video_name
            crops_out = out_root / "crops" / manipulation_dir.name / video_name
            crops_tmp = out_root / "crops" / manipulation_dir.name / (video_name + "_processing")

            # Skip if already logged as processed
            if rel_video_name in processed_videos:
                tqdm.write(f"Skipping {rel_video_name} (logged processed).")
                # Optionally delete frames if cleanup_frames True and frames_out exists
                if cleanup_frames:
                    try_delete_frames(frames_out)
                continue

            # If final crops exist and meet min_crops, log and skip
            existing = count_jpgs(crops_out)
            if existing >= min_crops:
                tqdm.write(f"Skipping {rel_video_name} (found {existing} crops). Logging and continuing.")
                with open(log_path, "a") as f:
                    f.write(rel_video_name + "\n")
                processed_videos.add(rel_video_name)
                # Optionally delete frames for this video
                if cleanup_frames:
                    try_delete_frames(frames_out)
                continue

            # If a leftover tmp dir exists from a previous crash:
            if crops_tmp.exists():
                tmp_count = count_jpgs(crops_tmp)
                if tmp_count >= min_crops:
                    # finalize it by moving to final location (only if final doesn't already have enough)
                    if not crops_out.exists() or count_jpgs(crops_out) < min_crops:
                        if crops_out.exists():
                            shutil.rmtree(crops_out)
                        safe_move_dir(crops_tmp, crops_out)
                        tqdm.write(f"Recovered and finalized {rel_video_name} from tmp ({tmp_count} crops).")
                        with open(log_path, "a") as f:
                            f.write(rel_video_name + "\n")
                        processed_videos.add(rel_video_name)
                        # build and append metadata now
                        rows = build_metadata_rows_for_video(frames_out, crops_out, video_name, manipulation_dir.name, label)
                        append_metadata_rows(out_root, rows)
                        # optionally delete frames after recovery
                        if cleanup_frames:
                            try_delete_frames(frames_out)
                        continue
                    else:
                        # final already ok, remove tmp
                        shutil.rmtree(crops_tmp)
                        tqdm.write(f"Found tmp for {rel_video_name} but final already OK. Removed tmp.")
                        with open(log_path, "a") as f:
                            f.write(rel_video_name + "\n")
                        processed_videos.add(rel_video_name)
                        if cleanup_frames:
                            try_delete_frames(frames_out)
                        continue
                else:
                    # tmp exists but too small -> remove and reprocess
                    shutil.rmtree(crops_tmp)
                    tqdm.write(f"Removed small tmp for {rel_video_name} ({tmp_count} crops). Will reprocess.")

            # 1) extract frames (sampled) if not present
            try:
                if not any(frames_out.glob("*.jpg")):
                    tqdm.write(f"Extracting frames for {rel_video_name} ...")
                    extract_frames(video_path, frames_out, step=frames_step)
            except subprocess.CalledProcessError as e:
                tqdm.write(f"ffmpeg failed for {video_path}: {e}")
                continue

            # 2) detect + save crops into tmp directory
            tqdm.write(f"Detecting faces for {rel_video_name} ...")
            crops_tmp.parent.mkdir(parents=True, exist_ok=True)
            saved = detect_and_save_crops(frames_out, crops_tmp, mtcnn, min_face_size=min_face_size, size=crop_size)

            # Decide whether to finalize
            tmp_count = count_jpgs(crops_tmp)
            if tmp_count >= min_crops:
                # If final crops dir exists but small, remove it
                if crops_out.exists() and count_jpgs(crops_out) < min_crops:
                    shutil.rmtree(crops_out)
                # Move tmp -> final
                safe_move_dir(crops_tmp, crops_out)
                tqdm.write(f"Finalized {rel_video_name}: {tmp_count} crops saved.")
                # Log processed video
                with open(log_path, "a") as f:
                    f.write(rel_video_name + "\n")
                processed_videos.add(rel_video_name)
                # Build metadata rows and append incrementally
                rows = build_metadata_rows_for_video(frames_out, crops_out, video_name, manipulation_dir.name, label)
                append_metadata_rows(out_root, rows)
                # Optionally delete frames to save space
                if cleanup_frames:
                    try_delete_frames(frames_out)
            else:
                # Not enough crops produced — remove tmp so next run retries
                tqdm.write(f"Only {tmp_count} crops found for {rel_video_name} (<{min_crops}). Deleting tmp and skipping (will retry later).")
                if crops_tmp.exists():
                    shutil.rmtree(crops_tmp)
                # Do NOT log the video so it can be retried later
                continue

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="data/raw/ffpp")
    parser.add_argument("--out_root", type=str, default="data/processed")
    parser.add_argument("--frames_step", type=int, default=3, help="keep every N-th frame")
    parser.add_argument("--crop_size", type=int, default=160)
    parser.add_argument("--min_face_size", type=int, default=40)
    parser.add_argument("--min_crops", type=int, default=5, help="minimum crops to consider a video successfully processed")
    parser.add_argument("--cleanup_frames", action="store_true", help="If set, delete the extracted frames/ directory for a video after finalizing crops (saves disk space).")
    args = parser.parse_args()
    process_ffpp(args.raw_root, args.out_root, frames_step=args.frames_step, crop_size=args.crop_size, min_face_size=args.min_face_size, min_crops=args.min_crops, cleanup_frames=args.cleanup_frames)
