#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ $# -lt 2 ]; then
  echo "Usage: $0 input_mesh output_image [resolution]"
  echo "Example: $0 ./models/chair.obj ./out/chair.png 1024"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"
RES="${3:-1024}"

# Call Blender in background and run render_mesh.py
blender --background --python "$ROOT_DIR/render_mesh.py" -- --input "$INPUT" --output "$OUTPUT" --res "$RES"
