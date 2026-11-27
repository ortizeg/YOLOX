#!/usr/bin/env bash
# On the training box: multi-GPU YOLOX training (Objects365 pretrain by default; EXP=... INIT=<ckpt> for the COCO
# fine-tune). Resumes from the newest latest_ckpt.pth (local, else the GCS run folder); otherwise starts from INIT.
# SYNC=gcs copies latest/best/log to $GR every 10 min.
#   RUN=<name> BPG=16 PERF=<flags> WORKERS=8 SYNC=gcs [EXP=.. INIT=.. GR=.. EXTRA="<exp opts>"] bash o365_train.sh
set -x
cd /workspace/YOLOX; export PYTHONPATH=/workspace/YOLOX
# one math-library thread per process: 8 ranks x (train + eval) workers x ~cores threads hit the container pids limit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN=${RUN:-o365_m_e20}; BPG=${BPG:-16}; WORKERS=${WORKERS:-8}; SYNC=${SYNC:-gcs}
GR=${GR:-${GCS_ROOT:-gs://YOUR_BUCKET/data}/objects365/runs/$RUN}
EXP=${EXP:-exps/perf/yolox_m_objects365_pretrain.py}
NG=$(nvidia-smi -L | wc -l); OUT=YOLOX_outputs/$RUN; mkdir -p $OUT
[ -s $OUT/latest_ckpt.pth ] || { [ "$SYNC" = gcs ] && gcloud storage cp $GR/latest_ckpt.pth $GR/best_ckpt.pth $OUT/ 2>/dev/null; }
RESUME=(); [ -n "${INIT:-}" ] && RESUME=(-c "$INIT")
[ -s $OUT/latest_ckpt.pth ] && RESUME=(--resume -c $OUT/latest_ckpt.pth)
if [ "$SYNC" = gcs ]; then
  ( while true; do sleep 600
      for f in latest_ckpt.pth best_ckpt.pth last_epoch_ckpt.pth train_log.txt; do
        [ -s $OUT/$f ] && gcloud storage cp -q $OUT/$f $GR/$f.tmp && gcloud storage mv -q $GR/$f.tmp $GR/$f; done
    done ) & SYNCER=$!
fi
echo "train start $(date -u +%FT%TZ) run $RUN gpus $NG bpg $BPG resume ${RESUME[*]:-none}" >> /workspace/finish.log
python3 -m yolox.tools.train -f $EXP -expn $RUN -d $NG -b $(( BPG * NG )) --fp16 "${RESUME[@]}" --logger tensorboard \
  data_num_workers $WORKERS $EXTRA perf "$PERF" > /workspace/o365_train_$(date +%s).log 2>&1
RC=$?
[ -n "${SYNCER:-}" ] && kill $SYNCER
[ "$SYNC" = gcs ] && for f in latest_ckpt.pth best_ckpt.pth last_epoch_ckpt.pth train_log.txt; do
  [ -s $OUT/$f ] && gcloud storage cp -q $OUT/$f $GR/$f; done
echo "train exit $RC $(date -u +%FT%TZ)" >> /workspace/finish.log
