#!/usr/bin/env bash
# Upload local GGUF packs to https://huggingface.co/audiohacking/uniflow-audio-gguf
# Weights stay out of git (dist/ and models/ are gitignored).
#
# DiT files must use HF GGUF quant tags in the filename:
#   dit-F16.gguf  dit-Q8_0.gguf  dit-Q4_0.gguf
set -euo pipefail

REPO_ID="${HF_REPO_ID:-audiohacking/uniflow-audio-gguf}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="${ROOT}/dist/hf"
VARIANT="${1:-uniflow-audio-v1.1-small}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: install Hugging Face CLI: pip install -U huggingface_hub[cli]" >&2
  exit 1
fi

WHOAMI="$(hf auth whoami 2>&1 || true)"
if echo "${WHOAMI}" | grep -qiE 'not logged|error|unauthorized'; then
  echo "error: not logged in to Hugging Face." >&2
  echo "  Run:  hf auth login" >&2
  echo "  Or:   export HF_TOKEN=hf_...   # write access to ${REPO_ID}" >&2
  exit 1
fi
echo "Logged in as: ${WHOAMI}"

PACK="${STAGING}/${VARIANT}"
if [[ ! -d "${PACK}" ]]; then
  echo "error: missing pack dir ${PACK}" >&2
  exit 1
fi

for f in t5_encoder.gguf vae.gguf instructions.gguf spiece.model; do
  if [[ ! -e "${PACK}/${f}" ]]; then
    echo "error: missing ${PACK}/${f}" >&2
    exit 1
  fi
done

shopt -s nullglob
DITS=("${PACK}"/dit-F16.gguf "${PACK}"/dit-Q8_0.gguf "${PACK}"/dit-Q4_0.gguf)
# Also accept any dit-*.gguf
EXTRA=("${PACK}"/dit-*.gguf)
FOUND=()
for f in "${DITS[@]}" "${EXTRA[@]}"; do
  [[ -e "${f}" ]] || continue
  # dedupe
  skip=0
  for g in "${FOUND[@]+"${FOUND[@]}"}"; do
    [[ "${f}" == "${g}" ]] && skip=1 && break
  done
  [[ ${skip} -eq 1 ]] && continue
  FOUND+=("${f}")
done
if [[ ${#FOUND[@]} -eq 0 ]]; then
  echo "error: no dit-<QUANT>.gguf in ${PACK}" >&2
  echo "  expected dit-F16.gguf / dit-Q8_0.gguf / dit-Q4_0.gguf" >&2
  exit 1
fi
echo "DiT files:"
ls -lh "${FOUND[@]}"

echo "Creating repo if needed: ${REPO_ID}"
hf repo create "${REPO_ID}" --type model --private false 2>/dev/null || true

echo "Uploading model card..."
hf upload "${REPO_ID}" "${STAGING}/README.md" README.md

echo "Uploading ${VARIANT}/ ..."
hf upload "${REPO_ID}" "${PACK}" "${VARIANT}"

# Drop legacy untagged dit.gguf on Hub once tagged files are uploaded.
echo "Removing legacy ${VARIANT}/dit.gguf from Hub (if present)..."
hf repos delete-files "${REPO_ID}" "${VARIANT}/dit.gguf" 2>/dev/null || true

echo "Done: https://huggingface.co/${REPO_ID}/tree/main/${VARIANT}"
