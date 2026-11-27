#!/usr/bin/env bash
# Follow-up on the bench box: clean GPU check against a true baseline, then the end-to-end pair with the
# accuracy-validated flag set (no compile: ~5 min recompile per multiscale size).
set -x
cd /workspace/YOLOX; export PYTHONPATH=/workspace/YOLOX; O=/workspace/perf
EXP=exps/default/yolox_m_basketball_merged5_adamw.py
E1=(input_size 800,800 test_size 640,640 multiscale_range 5 basic_lr_per_img 2.5e-05 mixup_prob 0.5)
SET=nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map
python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --check --warmup 0 --iters 60 --out $O/check2.json \
  --sets no_empty_cache ema_foreach tierA simota_batch sppf $SET -- "${E1[@]}" > $O/check2.log 2>&1
rm -f $O/e2e2_seconds.txt
for MODE in base perf; do
  P=(); [ $MODE = perf ] && P=(perf $SET)
  T0=$(date +%s)
  python3 -m yolox.tools.train -f $EXP -expn e2e2_$MODE -d 1 -b 16 --fp16 -c /workspace/yolox_m.pth --cache ram -o \
    --logger tensorboard seed 0 "${E1[@]}" max_epoch 20 no_aug_epochs 3 warmup_epochs 2 eval_interval 5 "${P[@]}" \
    > $O/e2e2_$MODE.log 2>&1
  echo "$MODE $(( $(date +%s) - T0 ))" >> $O/e2e2_seconds.txt
  cp YOLOX_outputs/e2e2_$MODE/train_log.txt $O/e2e2_${MODE}_train_log.txt; rm -rf YOLOX_outputs/e2e2_$MODE/*.pth
done
echo "RERUN DONE" >> /workspace/finish.log
