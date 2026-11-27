#!/usr/bin/env bash
# Pull one taxonomy-ablation run from the GPU box and verify bytes.
# Usage: scripts/pull_taxonomy_run.sh <tag> <ssh_host> <ssh_port>
set -euo pipefail
tag=$1; host=$2; port=$3
E="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $HOME/.ssh/id_rsa -p $port"
dst=training_results/taxonomy_ablation/$tag
mkdir -p "$dst"
rsync -a -e "$E" \
    "root@$host:/workspace/YOLOX/YOLOX_outputs/$tag/best_ckpt.pth" \
    "root@$host:/workspace/YOLOX/YOLOX_outputs/$tag/train_log.txt" \
    "root@$host:/workspace/YOLOX/ablation_runs/$tag.cmd" \
    "root@$host:/workspace/YOLOX/ablation_runs/$tag.stdout" \
    "$dst/"
remote=$($E "root@$host" "sha256sum /workspace/YOLOX/YOLOX_outputs/$tag/best_ckpt.pth" | cut -d' ' -f1)
local_sum=$(shasum -a 256 "$dst/best_ckpt.pth" | cut -d' ' -f1)
[ "$remote" = "$local_sum" ] || { echo "SHA MISMATCH $tag"; exit 1; }
echo "$local_sum  best_ckpt.pth" > "$dst/best_ckpt.sha256"
echo "pulled $tag ok $local_sum"
