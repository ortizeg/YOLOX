#!/usr/bin/env bash
# Second pass on a fresh box (after setup_box.sh): tests, profile, eval check, end-to-end x3.
set -x
cd /workspace/YOLOX; export PYTHONPATH=/workspace/YOLOX; O=/workspace/perf2; mkdir -p $O
EXP=exps/default/yolox_m_basketball_merged5_adamw.py
E1=(input_size 800,800 test_size 640,640 multiscale_range 5 basic_lr_per_img 2.5e-05 mixup_prob 0.5)
V1=nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map
V2=$V1,async_ckpt,eval_persistent,eval_amp
pip install -q pytest 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv > $O/gpu.txt; nproc >> $O/gpu.txt
python3 -m pytest tests -q -p no:cacheprovider > $O/pytest.txt 2>&1
python3 tools/perf/profile_step.py -f $EXP -c /workspace/yolox_m.pth --flags $V1 --out $O/profile.json -- "${E1[@]}" > $O/profile.log 2>&1
rm -f $O/e2e_seconds.txt
for MODE in base v1 v2; do
  P=(); [ $MODE = v1 ] && P=(perf $V1); [ $MODE = v2 ] && P=(perf $V2)
  T0=$(date +%s)
  python3 -m yolox.tools.train -f $EXP -expn p2_$MODE -d 1 -b 16 --fp16 -c /workspace/yolox_m.pth --cache ram -o \
    --logger tensorboard seed 0 "${E1[@]}" max_epoch 20 no_aug_epochs 3 warmup_epochs 2 eval_interval 5 "${P[@]}" \
    > $O/e2e_$MODE.log 2>&1
  echo "$MODE $(( $(date +%s) - T0 ))" >> $O/e2e_seconds.txt
  cp YOLOX_outputs/p2_$MODE/train_log.txt $O/e2e_${MODE}_train_log.txt
done
python3 tools/perf/eval_check.py -f $EXP -c YOLOX_outputs/p2_base/best_ckpt.pth --out $O/eval_check.json -- "${E1[@]}" > $O/eval_check.log 2>&1
rm -rf YOLOX_outputs/p2_*/*.pth
echo "PASS2 DONE" >> /workspace/finish.log
