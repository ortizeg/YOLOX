#!/usr/bin/env bash
# On the training box: COCO 2017 from our bucket into /workspace/YOLOX/datasets/COCO (YOLOX layout). "COCO PREP DONE".
set -euo pipefail
G=${GCS_ROOT:-gs://YOUR_BUCKET/data}/coco/coco2017
D=/workspace/YOLOX/datasets/COCO; mkdir -p $D/annotations
gcloud storage cp $G/train2017/annotations/instances_train2017.json $G/val2017/annotations/instances_val2017.json $D/annotations/
gcloud storage rsync -r -x 'annotations/.*' $G/val2017 $D/val2017
gcloud storage rsync -r -x 'annotations/.*' $G/train2017 $D/train2017
echo "train $(ls $D/train2017 | wc -l) val $(ls $D/val2017 | wc -l)" > /workspace/coco_prep.txt
echo "COCO PREP DONE $(date -u +%FT%TZ)" >> /workspace/finish.log
