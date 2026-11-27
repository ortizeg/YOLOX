#!/usr/bin/env bash
# On a vast box after foundation-finetune's setup_box.sh has unpacked this tree to /workspace/YOLOX, built
# datasets/basketball_merged5 and fetched yolox_m.pth. Writes everything to /workspace/perf and BENCH DONE to
# /workspace/finish.log. E1 recipe = the tuned 640 YOLOX-M arm (yolox_gap.md).
set -x
cd /workspace/YOLOX
export PYTHONPATH=/workspace/YOLOX
O=/workspace/perf; mkdir -p $O
EXP=exps/default/yolox_m_basketball_merged5_adamw.py
E1=(input_size 800,800 test_size 640,640 multiscale_range 5 basic_lr_per_img 2.5e-05 mixup_prob 0.5)
pip install -q pytest 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $O/gpu.txt; nproc >> $O/gpu.txt
python3 -c "import torch,cv2;print(torch.__version__,torch.version.cuda,cv2.__version__)" >> $O/gpu.txt

# 1. unit tests (CPU equivalence + existing suite) on the box's stack
[ -s $O/pytest.txt ] || python3 -m pytest tests -q -p no:cacheprovider > $O/pytest.txt 2>&1
# 2. GPU equivalence: deterministic algorithms, losses per iteration + final EMA weights vs baseline
[ -s $O/check.json ] || python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --check --warmup 0 --iters 60 --out $O/check.json \
  --sets no_empty_cache simota_batch ema_foreach sppf tierA cls_cost_algebra fused_optim channels_last focus_conv compile all \
  -- "${E1[@]}" > $O/check.log 2>&1
# 3. GPU-step timing: cumulative stack + single flags for attribution
C1=nondeterministic; C2=$C1,no_empty_cache; C3=$C2,simota_batch; C4=$C3,cls_cost_algebra; C5=$C4,ema_foreach
C6=$C5,fused_optim; C7=$C6,sppf,focus_conv; C8=$C7,channels_last; C9=$C8,compile
[ -s $O/step.json ] || python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --out $O/step.json \
  --sets "" $C1 $C2 $C3 $C4 $C5 $C6 $C7 $C8 $C9 no_empty_cache simota_batch channels_last compile \
  -- "${E1[@]}" > $O/step.log 2>&1
# 4. data path
[ -s $O/data.json ] || python3 tools/perf/bench_data.py -f $EXP --workers 4 8 --batches 60 --out $O/data.json "${E1[@]}" > $O/data.log 2>&1
# 5. end to end, same box, back to back: 20 epochs of the real training command (eval every 5, cache ram)
rm -f $O/e2e_seconds.txt
for MODE in base perf; do
  P=(); [ $MODE = perf ] && P=(perf "${PERF_SET:-all}")
  T0=$(date +%s)
  python3 -m yolox.tools.train -f $EXP -expn e2e_$MODE -d 1 -b 16 --fp16 -c /workspace/yolox_m.pth --cache ram -o \
    --logger tensorboard seed 0 "${E1[@]}" max_epoch 20 no_aug_epochs 3 warmup_epochs 2 eval_interval 5 "${P[@]}" \
    > $O/e2e_$MODE.log 2>&1
  echo "$MODE $(( $(date +%s) - T0 ))" >> $O/e2e_seconds.txt
  cp YOLOX_outputs/e2e_$MODE/train_log.txt $O/e2e_${MODE}_train_log.txt 2>/dev/null
  rm -rf YOLOX_outputs/e2e_$MODE/*.pth
done
echo "BENCH DONE $(date -u +%FT%TZ)" >> /workspace/finish.log
