#!/usr/bin/env bash
set -euo pipefail

# default: run the webcam demo with onnx model path if provided
# usage examples:
# docker run --gpus all --device /dev/video0 ... image -- demo
# or override to run any command

if [ "${1:-}" = "demo" ]; then
  ONNX=${2:-/workspace/checkpoints/onnx/mobilenet_v2.onnx}
  echo "Starting webcam demo with ONNX: $ONNX"
  python src/webcam_demo.py --onnx "$ONNX" --use_cuda
else
  # run passed command or a bash shell
  if [ "$#" -gt 0 ]; then
    exec "$@"
  else
    exec /bin/bash
  fi
fi
