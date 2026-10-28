#!/usr/bin/env bash
# Pull one YOLO26m taxonomy-ablation run (after export) and verify bytes.
# Usage: scripts/pull_yolo26_run.sh <tag> <ssh_host> <ssh_port>
set -euo pipefail
tag=$1; host=$2; port=$3
E="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $HOME/.ssh/id_rsa -p $port"
src=/workspace/yolo26_runs
dst=training_results/taxonomy_ablation/yolo26m/$tag
mkdir -p "$dst"
rsync -a -e "$E" \
    "root@$host:$src/$tag/weights/best.pt" \
    "root@$host:$src/$tag/yolo26m_basketball_${tag}_640.onnx" \
    "root@$host:$src/$tag/labels_mapping.json" \
    "root@$host:$src/$tag/results.csv" \
    "root@$host:$src/$tag/args.yaml" \
    "root@$host:$src/logs/$tag.py" \
    "root@$host:$src/logs/$tag.stdout" \
    "root@$host:$src/logs/$tag.export.log" \
    "$dst/"
for f in best.pt "yolo26m_basketball_${tag}_640.onnx"; do
    remote=$($E "root@$host" "sha256sum $src/$tag/$( [ $f = best.pt ] && echo weights/ )$f" | cut -d' ' -f1)
    local_sum=$(shasum -a 256 "$dst/$f" | cut -d' ' -f1)
    [ "$remote" = "$local_sum" ] || { echo "SHA MISMATCH $tag $f"; exit 1; }
    echo "$local_sum  $f" >> "$dst/sha256.txt"
done
best=$(python3 -c "
import csv; r=list(csv.DictReader(open('$dst/results.csv'))); r=[{k.strip():v for k,v in x.items()} for x in r]
b=max(r,key=lambda x: float(x['metrics/mAP50-95(B)']))  # 8.4.21 fitness = mAP50-95
print('best(fitness) epoch', b['epoch'], 'val mAP50-95', b['metrics/mAP50-95(B)'], 'epochs run', len(r))")
echo "pulled $tag ok; $best"
