#!/bin/sh
# Nesne tanima modeli: YOLO11n 320 ONNX (AGPL-3.0, Ultralytics). Git'te tutulmaz, buradan indirilir.
set -e
cd "$(dirname "$0")/.."
mkdir -p models
URL=https://huggingface.co/giangndm/yolo11-onnx/resolve/main/yolo11n_320.onnx
SHA=ed25687f7f3176a041bcdd0173501629afcaaddf37711f2294ced377f249fd65
curl -sSfL -o models/yolo11n_320.onnx "$URL"
echo "$SHA  models/yolo11n_320.onnx" | sha256sum -c -
