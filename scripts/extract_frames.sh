#!/usr/bin/env bash
# Usage: ./extract_frames.sh /path/to/ffpp /path/to/output_frames 5
FFPP_ROOT=${1:-"data/raw/ffpp"}
OUTROOT=${2:-"data/frames"}
STEP=${3:-5}   # extract every STEP frames

mkdir -p "$OUTROOT"
find "$FFPP_ROOT" -type f -iname "*.mp4" | while read vid; do
  # create a folder per video (base name without extension)
  base=$(basename "$vid")
  name="${base%.*}"
  outdir="$OUTROOT/$name"
  mkdir -p "$outdir"
  # ffmpeg: -vf "select=not(mod(n\,STEP))" ; -vsync 0 to keep frames numbering
  ffmpeg -i "$vid" -vf "select=not(mod(n\,$STEP))" -vsync 0 -qscale:v 2 "$outdir/frame_%06d.jpg"
done
