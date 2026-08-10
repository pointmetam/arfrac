#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/arface"
MODEL_PATH="$MODEL_DIR/face_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

mkdir -p "$MODEL_DIR"
curl --fail --location --retry 5 --retry-delay 2 "$MODEL_URL" --output "$MODEL_PATH"
echo "Downloaded $MODEL_PATH"
