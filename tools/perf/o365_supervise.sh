#!/usr/bin/env bash
# Overnight supervisor on the H200 box: resume Objects365 on a crash, then the COCO fine-tune from its best_ckpt
# (resumed on a crash too). In-training eval is off (8-GPU eval->train transition hung at epoch 4); each stage
# is scored afterwards by a single-GPU tools/eval.py on its final latest_ckpt.pth. Stage outcome = exit code of the last "train exit" after the last "train start" in
# finish.log ("none" when the process died without writing one).
V=nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map,ddp_bucket_view
F=$V,async_ckpt,eval_persistent,eval_amp,cv_flip,preproc_nopad,hwc_transport,int_indices,fast_anno,simota_nosync,resize_nosync,anno_cache,decode_reuse
O=/workspace/YOLOX/YOLOX_outputs
log() { echo "SUPERVISOR $* $(date -u +%FT%TZ)" >> /workspace/finish.log; }
last_exit() { awk '/train start/{e="none"} /train exit/{e=$3} END{print (e=="" ? "none" : e)}' /workspace/finish.log; }
running() { pgrep -f "[y]olox.tools.train" >/dev/null; }
wait_done() { while running; do sleep 60; done; sleep 30; while running; do sleep 60; done; }
launch_o365() { env EXTRA="eval_interval 1000 no_aug_eval_interval 1000" RUN=o365_m_e20 BPG=16 WORKERS=16 SYNC=none PERF=$F bash /workspace/YOLOX/tools/perf/o365_train.sh >> /workspace/o365_run.log 2>&1 & }
launch_coco() { env EXTRA="eval_interval 1000 no_aug_eval_interval 1000" RUN=coco_ft_o365_e80 EXP=exps/perf/yolox_m_coco_ft_o365.py INIT=/workspace/o365_best.pth BPG=16 WORKERS=16 \
  SYNC=none PERF=$F bash /workspace/YOLOX/tools/perf/o365_train.sh >> /workspace/coco_run.log 2>&1 & }

offline_eval() {  # exp ckpt name -> /workspace/eval_<name>.log, single GPU 0 (no DDP, original eval path)
  ( cd /workspace/YOLOX && CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=/workspace/YOLOX python3 -m yolox.tools.eval -f $1 -c $2       -b 32 -d 1 --conf 0.001 --fp16 > /workspace/eval_$3.log 2>&1
    log "eval $3 done: $(grep -E 'Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all' /workspace/eval_$3.log | head -1)" )
}

# stage 1: Objects365 (already running when the supervisor starts)
tries=0
while true; do
  wait_done
  rc=$(last_exit)
  [ "$rc" = 0 ] && { log "o365 complete"; break; }
  tries=$((tries+1)); [ $tries -gt 4 ] && { log "giving up on o365 (exit $rc)"; exit 1; }
  log "resume o365 (exit $rc) try $tries"; launch_o365; sleep 180
done

# stage 2: COCO fine-tune
until grep -q "COCO UNZIP DONE" /workspace/finish.log; do
  grep -q "COCO UNZIP FAILED" /workspace/finish.log && { log "coco unzip failed"; exit 1; }; sleep 60; done
cp $O/o365_m_e20/latest_ckpt.pth /workspace/o365_best.pth   # final epoch (EMA) weights
offline_eval exps/perf/yolox_m_objects365_pretrain.py /workspace/o365_best.pth o365_val5k   # before COCO: -o occupies GPU memory
tries=0
while true; do
  log "start/resume coco try $tries"; launch_coco; sleep 180
  wait_done
  rc=$(last_exit)
  [ "$rc" = 0 ] && { log "coco complete"; break; }
  tries=$((tries+1)); [ $tries -gt 4 ] && { log "giving up on coco (exit $rc)"; exit 1; }
done
offline_eval exps/perf/yolox_m_coco_ft_o365.py $O/coco_ft_o365_e80/latest_ckpt.pth coco_val2017
wait
log "ALL DONE"
