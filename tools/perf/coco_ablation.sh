#!/usr/bin/env bash
# COCO accuracy ablation on one 8-GPU box: each arm = the 80-epoch COCO fine-tune from the Objects365 checkpoint
# (exps/perf/yolox_m_coco_ft_o365.py, the recipe that gave 47.0) with the arm's head options, then a single-GPU
# val2017 eval of its final checkpoint (dual-head arms: the o2o branch NMS-free, and the o2m branch with NMS).
# In-training eval is off (8-GPU eval->train hang). Arms run one after another; before each, /workspace/YOLOX is
# updated to origin/main, so arms whose code lands later are picked up. A crashed arm resumes from its
# latest_ckpt.pth (3 tries). Writes /workspace/coco_abl/<arm>/ and "COCO ARM DONE <arm>" / "COCO ABLATION DONE".
#   ARMS="base:|ctiny:tiny_px 16" INIT=/workspace/o365.pth bash tools/perf/coco_ablation.sh
set -u
INIT=${INIT:-/workspace/o365.pth}
EXP=exps/perf/yolox_m_coco_ft_o365.py
O=/workspace/coco_abl; mkdir -p $O
PERF=${PERF:-nondeterministic,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map,ddp_bucket_view,async_ckpt,eval_persistent,eval_amp,cv_flip,preproc_nopad,hwc_transport,int_indices,fast_anno,resize_nosync,anno_cache,worker_cv2_threads,host_label_counts,lean_ckpt,fused_optim_amp,log_onesync,cuda_graphs}
log() { echo "$* $(date -u +%FT%TZ)" >> /workspace/finish.log; }

coco_eval() {  # arm opts nms out
  ( cd /workspace/YOLOX && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/YOLOX python3 -m yolox.tools.eval -f $EXP \
      -c $O/$1/latest_ckpt.pth -b 32 -d 1 --conf 0.001 --nms $3 --fp16 $2 > $O/$1/$4.log 2>&1 )
  log "COCO EVAL $1 $4: $(grep -m1 -E 'Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all' $O/$1/$4.log)"
}

IFS='|' read -r -a LIST <<< "${ARMS:?ARMS}"
for spec in "${LIST[@]}"; do
  arm=${spec%%:*}; opts=${spec#*:}
  [ -e $O/$arm/DONE ] && continue
  mkdir -p $O/$arm
  git -C /workspace/YOLOX fetch -q origin && git -C /workspace/YOLOX checkout -q origin/main
  git -C /workspace/YOLOX rev-parse HEAD > $O/$arm/commit.txt
  echo "$opts" > $O/$arm/opts.txt
  log "COCO ARM START $arm opts '$opts'"
  for try in 1 2 3; do
    env RUN=coco_$arm EXP=$EXP INIT=$INIT BPG=16 WORKERS=16 SYNC=none PERF="$PERF" \
      EXTRA="eval_interval 1000 no_aug_eval_interval 1000 $opts" bash /workspace/YOLOX/tools/perf/o365_train.sh \
      >> $O/$arm/run.log 2>&1
    pkill -9 -f "[m]ultiprocessing.spawn"; sleep 10  # leftover ranks / loader workers of a crashed run
    grep -q "Training of experiment is done" /workspace/YOLOX/YOLOX_outputs/coco_$arm/train_log.txt 2>/dev/null && break
    log "COCO ARM RETRY $arm try $try"
  done
  R=/workspace/YOLOX/YOLOX_outputs/coco_$arm
  cp $R/latest_ckpt.pth $R/train_log.txt $O/$arm/ 2>/dev/null
  if [[ " $opts " == *" dual_head 1 "* ]]; then
    coco_eval $arm "$opts" 1.0 eval_o2o
    coco_eval $arm "$opts infer_o2o 0" 0.65 eval_o2m
  else
    coco_eval $arm "$opts" 0.65 eval
  fi
  touch $O/$arm/DONE; log "COCO ARM DONE $arm"
done
log "COCO ABLATION DONE"
