#!/usr/bin/env bash
# Run all Phase 4 validation on vast.ai instance
# Usage: run from LOCAL machine. It will SSH into the instance and execute everything.
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

echo "=== Phase 4.1: Baseline benchmarks (original code) ==="

# Copy benchmark script to the instance
scp_to scripts/benchmark_profile.py /workspace/YOLOX/scripts/benchmark_profile.py

# Run baseline benchmark with original code (single GPU)
ssh_run 'cd /workspace/YOLOX && mkdir -p benchmarks scripts && YOLOX_DATADIR=/workspace/datasets CUDA_VISIBLE_DEVICES=0 python3 scripts/benchmark_profile.py --output benchmarks/baseline_profile.json --batch-size 64 --train-iters 100 --infer-images 1000 --warmup 10'

# Fetch baseline results
mkdir -p benchmarks
scp_from /workspace/YOLOX/benchmarks/baseline_profile.json benchmarks/baseline_profile.json
echo "Baseline results saved to benchmarks/baseline_profile.json"

echo ""
echo "=== Phase 4.2: Deploy modernized code ==="

# Push the modernized branch to the fork (it's already pushed)
# Clone it on the instance alongside the original
ssh_run 'cd /workspace && if [ ! -d YOLOX-modernized ]; then git clone --branch dev/modernize-yolox https://github.com/ortizeg/YOLOX.git YOLOX-modernized; else cd YOLOX-modernized && git pull; fi'

# Install modernized YOLOX
ssh_run 'cd /workspace/YOLOX-modernized && pip3 install -e . 2>&1 | tail -5'

# Symlink datasets if not present
ssh_run 'cd /workspace/YOLOX-modernized && ln -sf /workspace/datasets datasets 2>/dev/null; ls -la datasets/'

echo ""
echo "=== Phase 4.2: Modernized benchmarks ==="

# Run modernized benchmark
ssh_run 'cd /workspace/YOLOX-modernized && mkdir -p benchmarks && YOLOX_DATADIR=/workspace/datasets CUDA_VISIBLE_DEVICES=0 python3 scripts/benchmark_profile.py --output benchmarks/modernized_profile.json --batch-size 64 --train-iters 100 --infer-images 1000 --warmup 10'

# Fetch modernized results
scp_from /workspace/YOLOX-modernized/benchmarks/modernized_profile.json benchmarks/modernized_profile.json
echo "Modernized results saved to benchmarks/modernized_profile.json"

echo ""
echo "=== Phase 4.2: Compare baseline vs modernized ==="
python3 -c "
import json

with open('benchmarks/baseline_profile.json') as f:
    baseline = json.load(f)
with open('benchmarks/modernized_profile.json') as f:
    modernized = json.load(f)

print(f\"{'Metric':<40} {'Baseline':>12} {'Modernized':>12} {'Delta':>8}\")
print('-' * 76)

for section in ['training', 'inference']:
    print(f'\n  [{section.upper()}]')
    b = baseline[section]
    m = modernized[section]
    for key in b:
        bv = b[key]
        mv = m[key]
        if isinstance(bv, (int, float)) and isinstance(mv, (int, float)) and bv > 0:
            delta = (mv - bv) / bv * 100
            flag = ' !!!' if abs(delta) > 5 else ''
            print(f\"  {key:<38} {bv:>12.2f} {mv:>12.2f} {delta:>+7.1f}%{flag}\")
"

echo ""
echo "=== Phase 4.3: Convergence validation (10 epochs) ==="

# Run 10-epoch training convergence check
ssh_run 'cd /workspace/YOLOX-modernized && YOLOX_DATADIR=/workspace/datasets python3 scripts/validate_training.py --data-dir /workspace/datasets/COCO --epochs 10 --batch-size 64 --output benchmarks/training_convergence.csv --fp16 2>&1 | tail -30'

scp_from /workspace/YOLOX-modernized/benchmarks/training_convergence.csv benchmarks/training_convergence.csv
echo "Training convergence saved to benchmarks/training_convergence.csv"

echo ""
echo "=== Phase 4.3: Inference validation (mAP) ==="

# Run inference validation using the existing pretrained checkpoint
ssh_run 'cd /workspace/YOLOX-modernized && YOLOX_DATADIR=/workspace/datasets python3 scripts/validate_inference.py --data-dir /workspace/datasets/COCO --ckpt /workspace/YOLOX/YOLOX_outputs/yolox_s_run20_match/best_ckpt.pth --output benchmarks/inference_validation.json --batch-size 64 --fuse 2>&1 | tail -20'

scp_from /workspace/YOLOX-modernized/benchmarks/inference_validation.json benchmarks/inference_validation.json
echo "Inference validation saved to benchmarks/inference_validation.json"

echo ""
echo "=== ALL PHASE 4 VALIDATION COMPLETE ==="
echo "Results in benchmarks/"
ls -la benchmarks/
