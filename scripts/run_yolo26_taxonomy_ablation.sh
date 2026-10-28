#!/usr/bin/env bash
# Taxonomy ablation for YOLO26m: 10-class vs merged 5-class, seeds 0-2.
# Mirrors yolo26-basketball-training/scripts/train.py (yolo26m.pt, epochs 300,
# imgsz 640, batch 8, ultralytics 8.4.21) with two additions applied to BOTH arms:
#   - seed=N (published run used the default seed 0, deterministic=True)
#   - the optimizer that optimizer='auto' resolved to for the published 10-class
#     run, pinned so it does not change with nc: AdamW, lr0=0.000714, momentum=0.9,
#     warmup_bias_lr=0.0 (auto picks lr0 = 0.002*5/(4+nc), i.e. 0.001111 at nc=5)
#
# Usage: bash run_yolo26_taxonomy_ablation.sh [run_tag ...]
set -euo pipefail

PY=${PY:-/workspace/ul/bin/python}
DATA=${DATA:-/workspace/yolo26_data}
OUT=${OUT:-/workspace/yolo26_runs}
mkdir -p "$OUT/logs"

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
    RUNS=(10c_seed0 5c_seed0 10c_seed1 5c_seed1 10c_seed2 5c_seed2)
fi

for tag in "${RUNS[@]}"; do
    arm=${tag%%_*}
    seed=${tag##*_seed}
    if [ -f "$OUT/logs/$tag.done" ]; then
        echo "skip $tag (done)"
        continue
    fi
    cat > "$OUT/logs/$tag.py" <<EOF
from ultralytics import YOLO
YOLO("yolo26m.pt").train(
    data="$DATA/$arm/data.yaml", epochs=300, imgsz=640, batch=8,
    project="$OUT", name="$tag", exist_ok=True, verbose=True,
    seed=$seed, deterministic=True,
    optimizer="AdamW", lr0=0.000714, momentum=0.9, warmup_bias_lr=0.0,
)
EOF
    start=$(date +%s)
    echo "[$(date -u +%FT%TZ)] start $tag"
    (cd "$OUT" && "$PY" "logs/$tag.py") > "$OUT/logs/$tag.stdout" 2>&1 || true
    end=$(date +%s)
    if grep -q -E "Traceback|CUDA out of memory" "$OUT/logs/$tag.stdout" \
        || [ ! -f "$OUT/$tag/weights/best.pt" ]; then
        echo "[$(date -u +%FT%TZ)] FAILED $tag after $((end - start))s"
        touch "$OUT/logs/$tag.failed"
        exit 1
    fi
    echo -e "$tag\t$start\t$end\t$((end - start))" >> "$OUT/logs/timings.tsv"
    touch "$OUT/logs/$tag.done"
    echo "[$(date -u +%FT%TZ)] done $tag in $((end - start))s"
done
echo ALL_RUNS_DONE
