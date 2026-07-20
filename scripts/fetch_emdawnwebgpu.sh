#!/usr/bin/env bash
# Download a Dawn emdawnwebgpu_pkg compatible with ggml-webgpu's C++ API.
# Emscripten ≤4.0.12's built-in port (v20250531) lacks InstanceFeatureName /
# QueueWorkDone(status, StringView) and will fail to compile current ggml.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${EMDAWNWEBGPU_TAG:-v20251002.162335}"
DEST="${1:-$ROOT/third_party/emdawnwebgpu_pkg}"
ZIP_URL="https://github.com/google/dawn/releases/download/${TAG}/emdawnwebgpu_pkg-${TAG}.zip"

if [[ -f "$DEST/emdawnwebgpu.port.py" && -f "$DEST/.uniflow_tag" ]]; then
  if [[ "$(cat "$DEST/.uniflow_tag")" == "$TAG" ]]; then
    echo "emdawnwebgpu already present: $DEST ($TAG)"
    printf '%s\n' "$DEST"
    exit 0
  fi
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "Downloading emdawnwebgpu ${TAG}..."
curl -fsSL -o "$tmp/pkg.zip" "$ZIP_URL"
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
unzip -q "$tmp/pkg.zip" -d "$tmp/out"
if [[ -d "$tmp/out/emdawnwebgpu_pkg" ]]; then
  mv "$tmp/out/emdawnwebgpu_pkg" "$DEST"
else
  mv "$tmp/out" "$DEST"
fi
test -f "$DEST/emdawnwebgpu.port.py"
echo "$TAG" > "$DEST/.uniflow_tag"
echo "Installed: $DEST"
printf '%s\n' "$DEST"
