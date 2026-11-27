#!/usr/bin/env bash
# On the training box: Objects365 v2 train images + annotations into /workspace/YOLOX/datasets/objects365 and the
# train / val5k split jsons. SRC=gcs (our bucket; GCP VMs) or SRC=baai (public BAAI links; vast boxes, no egress fee).
# Streams each tar straight into tar -x (no tarball on disk), PAR patches at a time. Writes "O365 PREP DONE".
set -euo pipefail
SRC=${SRC:-gcs}; PAR=${PAR:-8}
G=${GCS_ROOT:-gs://YOUR_BUCKET/data}/objects365/raw/train
B=https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train
D=/workspace/YOLOX/datasets/objects365; mkdir -p $D/train $D/annotations /workspace/o365log
T0=$(date +%s)
fetch() {  # patch name -> extracted under $D/train
  local f=$1 m=/workspace/o365log/$1.done
  [ -f $m ] && return 0
  if [ "$SRC" = gcs ]; then gcloud storage cat "$G/$f.tar.gz" | tar -I pigz -x -C $D/train
  else
    for i in 1 2 3 4 5; do
      aria2c -q -x 16 -s 16 -k 16M -d /workspace/dl -o $f.tar.gz "$B/$f.tar.gz" && break; sleep 10; done
    tar -I pigz -xf /workspace/dl/$f.tar.gz -C $D/train && rm -f /workspace/dl/$f.tar.gz
  fi
  touch $m
}
export -f fetch; export SRC G B D
command -v pigz >/dev/null || (apt-get update -qq && apt-get install -y -qq pigz aria2 >/dev/null)
seq 0 50 | sed 's/^/patch/' | xargs -P $PAR -I{} bash -c 'fetch {} || echo "FETCH FAILED {}" >> /workspace/finish.log'
mkdir -p /workspace/o365ann
if [ "$SRC" = gcs ]; then gcloud storage cp "$G/zhiyuan_objv2_train_extracted/**" /workspace/o365ann/ 2>/dev/null || \
  gcloud storage cp -r "$G/zhiyuan_objv2_train_extracted" /workspace/o365ann
else mkdir -p /workspace/o365ann && aria2c -q -x 16 -s 16 -d /workspace/dl -o ann.tar.gz "$B/zhiyuan_objv2_train.tar.gz" && \
  tar -I pigz -xf /workspace/dl/ann.tar.gz -C /workspace/o365ann; fi
echo "fetch_seconds $(( $(date +%s) - T0 ))" > /workspace/o365log/prep.txt
cd /workspace/YOLOX && python3 - <<'PY' >> /workspace/o365log/prep.txt 2>&1
import glob, json, os, random, time
D = "/workspace/YOLOX/datasets/objects365"
t = time.time()
have = {}
for root, _, files in os.walk(f"{D}/train"):
    for f in files:
        if f.endswith(".jpg"):
            have[f] = os.path.relpath(os.path.join(root, f), f"{D}/train")
js = json.load(open(glob.glob("/workspace/o365ann/**/zhiyuan_objv2_train.json", recursive=True)[0]))
print("images on disk", len(have), "json images", len(js["images"]), "annotations", len(js["annotations"]))
imgs = []
for im in js["images"]:
    b = os.path.basename(im["file_name"])
    if b in have:
        imgs.append(dict(im, file_name=have[b]))
print("matched images", len(imgs), "missing", len(js["images"]) - len(imgs))
random.Random(0).shuffle(imgs)
val, train = imgs[:5000], imgs[5000:]
vid = {im["id"] for im in val}
tid = {im["id"] for im in train}
cats = js["categories"]
va, ta = [], []
for a in js["annotations"]:
    (va if a["image_id"] in vid else ta if a["image_id"] in tid else []).append(a)
json.dump({"images": train, "annotations": ta, "categories": cats}, open(f"{D}/annotations/o365_train.json", "w"))
json.dump({"images": val, "annotations": va, "categories": cats}, open(f"{D}/annotations/o365_val5k.json", "w"))
print("train", len(train), len(ta), "val", len(val), len(va), "categories", len(cats), "split_seconds", round(time.time() - t))
PY
echo "O365 PREP DONE $(date -u +%FT%TZ)" >> /workspace/finish.log
