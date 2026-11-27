#!/usr/bin/env bash
# Local: rent one multi-GPU vast box, push only this source tree (~5 MB), run box_o365_scale.sh.
#   tools/perf/launch_o365_scale.sh <label> "<offer query>"
set -euo pipefail
L=$1; Q=$2
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
KEY=${FFM_SSH_KEY:-$HOME/.ssh/id_rsa}
TGZ=$(mktemp -t yolox_perf).tgz
tar -C "$HERE" --exclude=.git --exclude=.pixi --exclude=training_results --exclude=YOLOX_outputs --exclude='*.pth' \
  --exclude=./datasets --exclude=./docs -czf "$TGZ" .
OFFER=$(vastai search offers "$Q rentable=true verified=true disk_space>=120" -o dph --raw | python3 -c "import json,sys;o=json.load(sys.stdin);print(o[0]['id'] if o else '')")
[ -n "$OFFER" ] || { echo "no offer"; exit 2; }
ID=$(vastai create instance "$OFFER" --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 120 --ssh --direct --label "$L" --raw | python3 -c "import json,sys;print(json.load(sys.stdin)['new_contract'])")
echo "created $ID"
for _ in $(seq 1 90); do
  read -r ST H P < <(vastai show instance "$ID" --raw 2>/dev/null | python3 -c "import json,sys
try:
  i=json.load(sys.stdin);print(i.get('actual_status'),i.get('ssh_host'),i.get('ssh_port'))
except Exception: print('x - -')") || true
  if [ "$ST" = running ] && ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes -p "$P" root@$H true 2>/dev/null; then break; fi
  sleep 10
done
[ "$ST" = running ] || { echo "never came up; destroying"; vastai destroy instance "$ID" -y; exit 3; }
mkdir -p "$HERE/../../foundation-finetune/vast/instances"
printf 'INSTANCE_ID=%s\nSSH_HOST=%s\nSSH_PORT=%s\nSSH_KEY=%s\n' "$ID" "$H" "$P" "$KEY" > "$HERE/../../foundation-finetune/vast/instances/$L.env"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -p "$P" root@$H)
"${SSH[@]}" "touch ~/.no_auto_tmux; cat > /workspace/yolox_src.tgz" < "$TGZ"
"${SSH[@]}" "tar -xzOf /workspace/yolox_src.tgz ./tools/perf/box_o365_scale.sh > /workspace/box_o365_scale.sh && cd /workspace && (setsid nohup bash /workspace/box_o365_scale.sh > /workspace/o365_scale.log 2>&1 < /dev/null &) && echo STARTED $L"
