#!/bin/bash
# demo.sh — one-command build + run for CascadeGuard
#
# Usage:
#   export HF_TOKEN=hf_your_real_token_here
#   ./demo.sh
#
set -e

NOTEBOOK="sycophancy-induced-hallucination-in-llms__1___2_.ipynb"
IMAGE_NAME="cascadeguard"

# --- Sanity checks ---------------------------------------------------------
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set."
    echo "Run:  export HF_TOKEN=hf_your_real_token_here"
    echo "Do NOT rely on the hardcoded token inside the notebook — it is exposed"
    echo "and should be treated as compromised if this repo is public."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "ERROR: docker is not installed or not on PATH."
    exit 1
fi

if [ ! -f "$NOTEBOOK" ]; then
    echo "ERROR: $NOTEBOOK not found in current directory."
    exit 1
fi

# --- Build -------------------------------------------------------------
echo "==> Building Docker image ($IMAGE_NAME)..."
docker build -t "$IMAGE_NAME" .

# --- Pre-install missing deps up front (works around the notebook's own ---
# --- import-order bug: sentence-transformers is imported in an earlier   ---
# --- cell than the one that installs it)                                 ---
INSTALL_CMD="pip install -q datasets sentence-transformers scipy sentencepiece protobuf huggingface_hub"

# --- Run: execute the notebook headlessly inside the container ------------
echo "==> Running container and executing notebook end-to-end..."
docker run --gpus all --rm \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$IMAGE_NAME" \
    bash -c "$INSTALL_CMD && jupyter nbconvert --to notebook --execute --inplace '$NOTEBOOK'"

echo "==> Done. Outputs written into $NOTEBOOK."
