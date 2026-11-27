#!/usr/bin/env bash
# torch upgrade comparison on the N7 box (run after tools/perf/box_n7.sh): the image's torch 2.5.1+cu124 vs the newest
# release in a separate venv (TORCH_INDEX, default cu130), same code, same data. Writes /workspace/tup, "TUP DONE".
#   1. 1-GPU bench_step (E1 recipe, 110 warm-up + 220 timed), V and V+cuda_graphs, two orders, both stacks back to back
#   2. 2-GPU training step, V and V+cuda_graphs, 300 iterations: E1 (800 +-5x32, 16/GPU) and light (640, 8/GPU)
set -x
cd /workspace/YOLOX; export PYTHONPATH=/workspace/YOLOX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=/workspace/tup; mkdir -p $O
IDX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
NEW=/workspace/venv_new
if [ ! -x $NEW/bin/python ]; then
  python3 -m venv $NEW
  $NEW/bin/pip install -q torch torchvision --index-url $IDX > $O/pip.log 2>&1
  $NEW/bin/pip install -q loguru tabulate thop tqdm psutil tensorboard pycocotools opencv-python-headless numpy pytest >> $O/pip.log 2>&1
fi
EXP=exps/default/yolox_m_basketball_merged5_adamw.py
NG=$(nvidia-smi -L | wc -l)
V=$(grep "^V=" tools/perf/box_n7.sh | cut -d= -f2)
G=$V,cuda_graphs
E1=(input_size 800,800 test_size 640,640 multiscale_range 5 basic_lr_per_img 2.5e-05 mixup_prob 0.5)
for PY in python3 $NEW/bin/python; do
  $PY -c "import torch,cv2;print(torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),cv2.__version__)"
done > $O/env.txt 2>&1
# new stack: the N6/N7 tests first (does the code run unchanged?)
$NEW/bin/python -m pytest tests/perf/test_perf_n6_syncs.py tests/perf/test_perf_cuda_graphs.py tests/perf/test_perf_fused_amp.py -q -p no:cacheprovider > $O/pytest_new.txt 2>&1

for PY in ${STACKS:-python3 $NEW/bin/python}; do T=old; [ $PY = python3 ] || T=new
  $PY tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --warmup 110 --iters 220 --out $O/step_1gpu_$T.json \
    --sets "$V" "$G" "$V" "$G" -- "${E1[@]}" > $O/step_1gpu_$T.log 2>&1
done

train() {  # name python perf bpg "<extra>"
  local name=$1 py=$2 p=$3 bpg=$4 extra=$5
  YOLOX_AUDIT_ITERS=300 YOLOX_AUDIT_SPAN_EPOCHS=1 YOLOX_AUDIT_OUT=$O/mem_$name timeout -k 60 1800 \
    $py -m yolox.tools.train -f $EXP -expn tup_$name -d $NG -b $(( bpg * NG )) --fp16 -c /workspace/yolox_m.pth \
    --logger tensorboard seed 0 "${E1[@]}" max_epoch 300 no_aug_epochs 0 warmup_epochs 1 eval_interval 1000 \
    data_num_workers 8 print_interval 10 $extra perf "$p" > $O/$name.log 2>&1
  echo "$name rc=$?" >> $O/runs.txt
  sleep 5; pkill -9 -f "[y]olox.tools.train"; pkill -9 -f "[m]ultiprocessing.spawn"; sleep 10
  cp YOLOX_outputs/tup_$name/train_log.txt $O/train_$name.txt 2>/dev/null; rm -rf YOLOX_outputs/tup_$name
}
for S in V G; do P=$V; [ $S = G ] && P=$G
  train e1_${S}_new $NEW/bin/python "$P" 16 ""
  train light_${S}_new $NEW/bin/python "$P" 8 "input_size 640,640 multiscale_range 0"
done

python3 - > $O/digest.txt 2>&1 <<'PY'
import glob, json, os, re
O = "/workspace/tup"
print(open(f"{O}/env.txt").read())
for T in ["old", "new"]:
    f = f"{O}/step_1gpu_{T}.json"
    if os.path.exists(f):
        d = json.load(open(f))
        for i, (k, v) in enumerate(d["sets"].items()):
            print(f"1gpu {T} #{i} {'G' if 'cuda_graphs' in k else ('V' if k else 'baseline')}: "
                  f"{v['sec_per_iter_median']*1000:.1f} ms median")
for f in sorted(glob.glob(f"{O}/train_*.txt")):
    t = [float(x) for x in re.findall(r"iter_time: ([0-9.]+)s", open(f).read())][10:]
    print(f"{os.path.basename(f)}: mean iter_time {sum(t)/max(len(t),1)*1000:.1f} ms over {len(t)} log lines")
PY
echo "TUP DONE $(date -u +%FT%TZ)" >> /workspace/finish.log
