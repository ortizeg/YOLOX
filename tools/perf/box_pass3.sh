#!/usr/bin/env bash
# Pass-3 GPU check on a box that already ran setup_box.sh (tree refreshed in place).
set -x
cd /workspace/YOLOX; export PYTHONPATH=/workspace/YOLOX; O=/workspace/perf3; mkdir -p $O
EXP=exps/default/yolox_m_basketball_merged5_adamw.py
E1=(input_size 800,800 test_size 640,640 multiscale_range 5 basic_lr_per_img 2.5e-05 mixup_prob 0.5)
V=nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map
V2=$V,async_ckpt,eval_persistent,eval_amp
P3=cv_flip,preproc_nopad,hwc_transport,int_indices,fast_anno,simota_nosync,resize_nosync
python3 -m pytest tests -q -p no:cacheprovider > $O/pytest.txt 2>&1
python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --check --iters 60 --warmup 0 --ref "$V" \
  --out $O/check.json --sets "$V,simota_nosync" "$V,resize_nosync,int_indices" -- "${E1[@]}" > $O/check.log 2>&1
python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --out $O/step.json --sets "$V" "$V,simota_nosync" \
  -- "${E1[@]}" > $O/step.log 2>&1
python3 tools/perf/bench_data.py -f $EXP --workers 4 8 --batches 60 --out $O/data.json \
  --sets "" data "data,cv_flip,preproc_nopad,hwc_transport,int_indices" -- "${E1[@]}" > $O/data.log 2>&1
rm -f $O/e2e_seconds.txt
for MODE in v2 v3; do
  P=(perf $V2); [ $MODE = v3 ] && P=(perf $V2,$P3)
  T0=$(date +%s)
  python3 -m yolox.tools.train -f $EXP -expn p3_$MODE -d 1 -b 16 --fp16 -c /workspace/yolox_m.pth --cache ram -o \
    --logger tensorboard seed 0 "${E1[@]}" max_epoch 20 no_aug_epochs 3 warmup_epochs 2 eval_interval 5 "${P[@]}" \
    > $O/e2e_$MODE.log 2>&1
  echo "$MODE $(( $(date +%s) - T0 ))" >> $O/e2e_seconds.txt
  cp YOLOX_outputs/p3_$MODE/train_log.txt $O/e2e_${MODE}_train_log.txt; rm -rf YOLOX_outputs/p3_$MODE/*.pth
done
echo "PASS3 DONE" >> /workspace/finish.log
