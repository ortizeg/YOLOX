#!/usr/bin/env bash
# Local: rent one vast box, push this working tree + foundation-finetune, run setup_box.sh then box_bench.sh.
#   tools/perf/launch_bench.sh <label> "<offer query>"
set -euo pipefail
L=$1; Q=$2
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
FFM=${FFM:-"$HERE/../../foundation-finetune"}
TGZ=$(mktemp -t yolox_perf).tgz
tar -C "$HERE" --exclude=.git --exclude=.pixi --exclude=training_results --exclude=YOLOX_outputs \
  --exclude='*.pth' --exclude=datasets/basketball_merged5 -czf "$TGZ" .
env -u HF_KEY "$FFM/vast/launch.sh" "$L" "$Q" pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel 80
source "$FFM/vast/instances/$L.env"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -p "$SSH_PORT" "root@$SSH_HOST")
"${SSH[@]}" "cat > /workspace/yolox_src.tgz" < "$TGZ"
"${SSH[@]}" "cd /workspace && setsid nohup bash -c 'bash /workspace/ffm/models/yolox/setup_box.sh > /workspace/setup.log 2>&1 && bash /workspace/YOLOX/tools/perf/${BOX_SCRIPT:-box_bench.sh} > /workspace/bench.log 2>&1 || echo BENCH FAILED >> /workspace/finish.log' > /dev/null 2>&1 < /dev/null & sleep 2; echo STARTED $L"
