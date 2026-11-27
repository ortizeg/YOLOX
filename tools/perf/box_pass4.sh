#!/usr/bin/env bash
# Pass 4 on one basketball box (RTX 4090) that already ran setup_box.sh: GPU step at batch 16/24/32/48 and 20-epoch
# wall clock for B16, B32 and decode_reuse. Writes /workspace/perf4/* and "PASS4 DONE" to /workspace/finish.log.
set -x
cd /workspace/YOLOX; export PYTHONPATH=/workspace/YOLOX; O=/workspace/perf4; mkdir -p $O
EXP=exps/default/yolox_m_basketball_merged5_adamw.py
E1=(input_size 800,800 test_size 640,640 multiscale_range 5 basic_lr_per_img 2.5e-05 mixup_prob 0.5)
V=nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map
F=$V,async_ckpt,eval_persistent,eval_amp,cv_flip,preproc_nopad,hwc_transport,int_indices,fast_anno,simota_nosync,resize_nosync
python3 -m pytest tests -q -p no:cacheprovider > $O/pytest.txt 2>&1
for BS in 16 24 32 48; do
  python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --batch $BS --out $O/step_b$BS.json --sets "$F" \
    -- "${E1[@]}" > $O/step_b$BS.log 2>&1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $O/step_b$BS.log
done
rm -f $O/e2e_seconds.txt
for MODE in b16 b32 b16_reuse; do
  BS=16; P="$F"; [ $MODE = b32 ] && BS=32; [ $MODE = b16_reuse ] && P="$F,decode_reuse"
  T0=$(date +%s)
  python3 -m yolox.tools.train -f $EXP -expn p4_$MODE -d 1 -b $BS --fp16 -c /workspace/yolox_m.pth --cache ram -o \
    --logger tensorboard seed 0 "${E1[@]}" max_epoch 20 no_aug_epochs 3 warmup_epochs 2 eval_interval 5 perf "$P" \
    > $O/e2e_$MODE.log 2>&1
  echo "$MODE $(( $(date +%s) - T0 ))" >> $O/e2e_seconds.txt
  cp YOLOX_outputs/p4_$MODE/train_log.txt $O/e2e_${MODE}_train_log.txt; rm -rf YOLOX_outputs/p4_$MODE/*.pth
done
echo "PASS4 DONE" >> /workspace/finish.log
