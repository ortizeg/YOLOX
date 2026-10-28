#!/usr/bin/env bash
# Taxonomy ablation: YOLOX-M 10-class (published exp) vs merged 5-class, seeds 0-2.
# Runs on the GPU box from the repo root. Identical command to the published run
# except -f, the seed opt, and -expn (output folder only).
#
# Usage: bash scripts/run_taxonomy_ablation.sh [run_tag ...]
#   default runs: 10c_seed0 5c_seed0 10c_seed1 5c_seed1 10c_seed2 5c_seed2
set -euo pipefail

CKPT=${CKPT:-/workspace/yolox_m.pth}
OUT=${OUT:-ablation_runs}
mkdir -p "$OUT"

# save_history_ckpt (exp default True) writes a ~300MB epoch_N_ckpt.pth at every
# eval, ~20GB per run. They are write-only (selection uses best_ckpt.pth), so
# delete them in the background instead of changing the training command.
( while true; do
    find YOLOX_outputs -name 'epoch_*_ckpt.pth' -mmin +1 -delete 2>/dev/null || true
    sleep 60
  done ) &
CLEANER=$!
trap 'kill $CLEANER 2>/dev/null' EXIT

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
    RUNS=(10c_seed0 5c_seed0 10c_seed1 5c_seed1 10c_seed2 5c_seed2)
fi

for tag in "${RUNS[@]}"; do
    case "$tag" in
        10c_*) exp=exps/default/yolox_m_basketball_adamw.py ;;
        5c_*) exp=exps/default/yolox_m_basketball_merged5_adamw.py ;;
        *) echo "unknown run tag $tag" >&2; exit 1 ;;
    esac
    seed=${tag##*_seed}
    if [ -f "$OUT/$tag.done" ]; then
        echo "skip $tag (done)"
        continue
    fi

    cmd=(pixi run python -m yolox.tools.train -f "$exp" -expn "$tag" -d 1 -b 16 --fp16
         -c "$CKPT" --cache ram -o --logger tensorboard seed "$seed")
    echo "${cmd[*]}" > "$OUT/$tag.cmd"
    start=$(date +%s)
    echo "[$(date -u +%FT%TZ)] start $tag"
    "${cmd[@]}" > "$OUT/$tag.stdout" 2>&1
    end=$(date +%s)
    # yolox's @logger.catch swallows exceptions and exits 0, so check the log.
    if grep -q -E "Traceback|CUDA out of memory" "$OUT/$tag.stdout"; then
        echo "[$(date -u +%FT%TZ)] FAILED $tag after $((end - start))s"
        touch "$OUT/$tag.failed"
        exit 1
    fi
    echo -e "$tag\t$start\t$end\t$((end - start))" >> "$OUT/timings.tsv"
    touch "$OUT/$tag.done"
    echo "[$(date -u +%FT%TZ)] done $tag in $((end - start))s"
done
echo ALL_RUNS_DONE
