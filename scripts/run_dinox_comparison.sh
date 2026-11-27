#!/usr/bin/env bash
# Compare YOLOX-S vs DINOX-S training on vast.ai instance.
# Usage: run from LOCAL machine. SSH config matches run_vastai_validation.sh.
set -euo pipefail

HOST="ssh6.vast.ai"
PORT="19562"
KEY="$HOME/.ssh/id_rsa"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4"

ssh_run() {
    ssh -p "$PORT" $SSH_OPTS -i "$KEY" "root@$HOST" "$@"
}

scp_to() {
    scp -P "$PORT" $SSH_OPTS -i "$KEY" "$1" "root@$HOST:$2"
}

scp_from() {
    scp -P "$PORT" $SSH_OPTS -i "$KEY" "root@$HOST:$1" "$2"
}

echo "=== Deploy DINOX branch ==="

# Update the branch on the instance
ssh_run 'cd /workspace/YOLOX-modernized && git fetch origin && git checkout dev/dinox-phase1 && git pull origin dev/dinox-phase1'

# Reinstall to pick up new modules
ssh_run 'cd /workspace/YOLOX-modernized && pip3 install -e . 2>&1 | tail -5'

# Symlink datasets
ssh_run 'cd /workspace/YOLOX-modernized && ln -sf /workspace/datasets datasets 2>/dev/null || true'

echo ""
echo "=== Run YOLOX-S vs DINOX-S comparison (10 epochs) ==="

ssh_run 'cd /workspace/YOLOX-modernized && \
    YOLOX_DATADIR=/workspace/datasets \
    CUDA_VISIBLE_DEVICES=0 \
    python3 scripts/compare_dinox_training.py \
        --data-dir /workspace/datasets/COCO \
        --epochs 10 \
        --batch-size 64 \
        --output-dir benchmarks/dinox_comparison \
        --fp16 \
    2>&1 | tee benchmarks/dinox_comparison/training_log.txt'

echo ""
echo "=== Fetching results ==="

mkdir -p benchmarks/dinox_comparison
scp_from "/workspace/YOLOX-modernized/benchmarks/dinox_comparison/*" benchmarks/dinox_comparison/

echo ""
echo "=== Results saved to benchmarks/dinox_comparison/ ==="
ls -la benchmarks/dinox_comparison/
