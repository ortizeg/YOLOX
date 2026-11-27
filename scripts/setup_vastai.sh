#!/usr/bin/env bash
# Setup a fresh vast.ai instance for YOLOX/DINOX training.
# Usage: ssh into instance, then:
#   curl -sSL https://raw.githubusercontent.com/ortizeg/YOLOX/<BRANCH>/scripts/setup_vastai.sh | bash -s -- <BRANCH> [COCO_DIR]
#
# Or copy this script and run:
#   bash scripts/setup_vastai.sh <BRANCH> [COCO_DIR]
#
# Arguments:
#   BRANCH   - git branch to checkout (e.g. dev/dinox-phase1)
#   COCO_DIR - where to download COCO (default: /workspace/datasets/COCO)

set -euo pipefail

BRANCH="${1:?Usage: setup_vastai.sh <BRANCH> [COCO_DIR]}"
COCO_DIR="${2:-/workspace/datasets/COCO}"
REPO_DIR="/workspace/YOLOX-dinox"
LOG_DIR="/workspace/output"

echo "=== YOLOX vast.ai setup ==="
echo "Branch: $BRANCH"
echo "COCO:   $COCO_DIR"
echo "Repo:   $REPO_DIR"
echo ""

# ----------------------------------------------------------------
# 1. System packages
# ----------------------------------------------------------------
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq libxcb1 libgl1-mesa-glx libglib2.0-0 unzip tmux 2>&1 | tail -1
echo "  Done."

# ----------------------------------------------------------------
# 2. Python packages
# ----------------------------------------------------------------
echo "[2/5] Installing Python packages..."

# Check if torch is already installed (PyTorch Docker images have it)
if python3 -c "import torch; print(f'PyTorch {torch.__version__} already installed')" 2>/dev/null; then
    echo "  PyTorch found, installing remaining deps..."
    pip install -q opencv-python pycocotools loguru tabulate tensorboard thop \
        tqdm psutil onnx onnx-simplifier 2>&1 | tail -1
else
    echo "  Installing PyTorch + all deps..."
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1
    pip install -q opencv-python pycocotools loguru tabulate tensorboard thop \
        tqdm psutil onnx onnx-simplifier 2>&1 | tail -1
fi
echo "  Done."

# ----------------------------------------------------------------
# 3. Clone repository
# ----------------------------------------------------------------
echo "[3/5] Cloning repository..."
mkdir -p /workspace
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git reset --hard "origin/$BRANCH"
    echo "  Updated existing repo."
else
    git clone --branch "$BRANCH" https://github.com/ortizeg/YOLOX.git "$REPO_DIR"
    echo "  Cloned fresh."
fi

# Verify import works
cd "$REPO_DIR"
python3 -c "from yolox.models import YOLOX; print('  YOLOX import OK')"

# ----------------------------------------------------------------
# 4. Download COCO dataset
# ----------------------------------------------------------------
if [ -f "$COCO_DIR/.download_complete" ]; then
    echo "[4/5] COCO already downloaded."
else
    echo "[4/5] Downloading COCO dataset (this takes ~5-10 minutes)..."
    mkdir -p "$COCO_DIR"
    cd "$COCO_DIR"

    if [ ! -d "annotations" ]; then
        echo "  Downloading annotations..."
        wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip -q annotations_trainval2017.zip
        rm annotations_trainval2017.zip
    fi

    if [ ! -d "train2017" ]; then
        echo "  Downloading train2017 (~18GB)..."
        wget -q http://images.cocodataset.org/zips/train2017.zip
        unzip -q train2017.zip
        rm train2017.zip
    fi

    if [ ! -d "val2017" ]; then
        echo "  Downloading val2017 (~1GB)..."
        wget -q http://images.cocodataset.org/zips/val2017.zip
        unzip -q val2017.zip
        rm val2017.zip
    fi

    touch .download_complete
    echo "  Done: $(ls train2017/ | wc -l) train, $(ls val2017/ | wc -l) val images."
fi

# Symlink datasets into repo
cd "$REPO_DIR"
ln -sf "$(dirname "$COCO_DIR")" datasets 2>/dev/null || true

# ----------------------------------------------------------------
# 5. Verify GPU setup
# ----------------------------------------------------------------
echo "[5/5] Verifying GPU setup..."
python3 -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem // (1024**3)
    print(f'  GPU {i}: {name} ({mem}GB)')
print(f'  {n} GPUs ready.')
"

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
mkdir -p "$LOG_DIR"
echo ""
echo "=== Setup complete ==="
echo "Repo:     $REPO_DIR"
echo "COCO:     $COCO_DIR"
echo "Logs:     $LOG_DIR"
echo ""
echo "To start training:"
echo "  cd $REPO_DIR"
echo "  export YOLOX_DATADIR=$(dirname $COCO_DIR)"
echo "  python3 -m yolox.tools.train -f exps/custom/YOUR_EXP.py -d NUM_GPUS -b BATCH --fp16 -o"
