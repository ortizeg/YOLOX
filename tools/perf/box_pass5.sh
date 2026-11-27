#!/usr/bin/env bash
# GPU check of the five pass-5 flags. Writes /workspace/p5 and "P5 DONE" to /workspace/finish.log.
set -x
cd /workspace/YOLOX; export PYTHONPATH=/workspace/YOLOX
O=/workspace/p5; mkdir -p $O
pip install -q pytest 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $O/gpu.txt; nproc >> $O/gpu.txt
python3 -c "import torch,cv2;print(torch.__version__,torch.version.cuda,cv2.__version__)" >> $O/gpu.txt
EXP=exps/default/yolox_m_basketball_merged5_adamw.py
cat > exps/perf/_bench_sgd.py <<'P'
# bench only: the merged5 basketball exp with the base (SGD, nesterov, 3 groups) optimizer, as in Objects365/COCO
import importlib.util, os
from yolox.exp.yolox_base import Exp as BaseExp
_p = os.path.join(os.path.dirname(__file__), "..", "default", "yolox_m_basketball_merged5_adamw.py")
_s = importlib.util.spec_from_file_location("bb", _p); _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
class Exp(_m.Exp):
    def get_optimizer(self, batch_size):
        return BaseExp.get_optimizer(self, batch_size)
P
E1=(input_size 800,800 test_size 640,640 multiscale_range 5 basic_lr_per_img 2.5e-05 mixup_prob 0.5)
V=$(python3 -c "from yolox.utils import perf;new={'head_nosync','worker_cv2_threads','host_label_counts','lean_ckpt','fused_optim_amp','compile'};print(','.join(sorted(perf.parse('all')-new)))")
A5=$V,head_nosync,worker_cv2_threads,host_label_counts,lean_ckpt,fused_optim_amp
echo "$V" > $O/V.txt
# 1. tests on GPU
python3 -m pytest tests/perf -q -p no:cacheprovider -rs > $O/pytest_perf.txt 2>&1
python3 -m pytest tests/perf/test_perf_head_nosync.py tests/perf/test_perf_host_counts.py tests/perf/test_perf_fused_amp.py tests/perf/test_perf_cv2_threads.py tests/perf/test_perf_lean_ckpt.py -v -p no:cacheprovider -rs > $O/pytest_new_v.txt 2>&1
# 2. equivalence
python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --check --warmup 0 --iters 60 --ref "$V" --out $O/check_adamw.json \
  --sets "$V,head_nosync" "$V,host_label_counts" "$V,fused_optim_amp" "$A5" -- "${E1[@]}" > $O/check_adamw.log 2>&1
python3 tools/perf/bench_step.py -f exps/perf/_bench_sgd.py -c /workspace/yolox_m.pth --check --warmup 0 --iters 60 --ref "$V" --out $O/check_sgd.json \
  --sets "$V,fused_optim_amp" "$A5" -- "${E1[@]}" > $O/check_sgd.log 2>&1
# 3. timing, two orders, AdamW and SGD
for R in 1 2; do
  if [ $R = 1 ]; then SETS=("$V" "$V,head_nosync" "$V,host_label_counts" "$V,fused_optim_amp" "$A5"); else SETS=("$A5" "$V,fused_optim_amp" "$V,host_label_counts" "$V,head_nosync" "$V"); fi
  python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --warmup 110 --iters 220 --out $O/step_adamw_r$R.json --sets "${SETS[@]}" -- "${E1[@]}" > $O/step_adamw_r$R.log 2>&1
done
python3 tools/perf/bench_step.py -f exps/perf/_bench_sgd.py -c /workspace/yolox_m.pth --warmup 110 --iters 220 --out $O/step_sgd.json --sets "$V" "$V,fused_optim_amp" "$A5" -- "${E1[@]}" > $O/step_sgd.log 2>&1
# 4. N9 end to end, 20 epochs, back to back
rm -f $O/e2e_seconds.txt
for MODE in base lean; do
  P=(perf "$V"); [ $MODE = lean ] && P=(perf "$V,lean_ckpt" save_interval 5)
  T0=$(date +%s)
  python3 -m yolox.tools.train -f $EXP -expn e2e_$MODE -d 1 -b 16 --fp16 -c /workspace/yolox_m.pth --cache ram \
    --logger tensorboard seed 0 "${E1[@]}" max_epoch 20 no_aug_epochs 3 warmup_epochs 2 eval_interval 5 "${P[@]}" > $O/e2e_$MODE.log 2>&1; RC=$?
  echo "$MODE $(( $(date +%s) - T0 )) rc=$RC" >> $O/e2e_seconds.txt
  ls -la --time-style=+%T YOLOX_outputs/e2e_$MODE/ > $O/e2e_${MODE}_files.txt
  du -sh YOLOX_outputs/e2e_$MODE >> $O/e2e_${MODE}_files.txt
  cp YOLOX_outputs/e2e_$MODE/train_log.txt $O/e2e_${MODE}_train_log.txt
done
python3 -c "import torch;c=torch.load('YOLOX_outputs/e2e_lean/latest_ckpt.pth',map_location='cpu',weights_only=False);print('latest keys',sorted(c),'start_epoch',c.get('start_epoch'))
for f in ['best_ckpt.pth','last_epoch_ckpt.pth']:
  import os
  p='YOLOX_outputs/e2e_lean/'+f
  if os.path.exists(p):
    c=torch.load(p,map_location='cpu',weights_only=False);print(f,sorted(c),os.path.getsize(p),os.stat(p).st_nlink)" > $O/lean_keys.txt 2>&1
T0=$(date +%s)
python3 -m yolox.tools.train -f $EXP -expn e2e_lean -d 1 -b 16 --fp16 --resume -c YOLOX_outputs/e2e_lean/latest_ckpt.pth --cache ram \
  --logger tensorboard seed 0 "${E1[@]}" max_epoch 21 no_aug_epochs 3 warmup_epochs 2 eval_interval 5 perf "$V,lean_ckpt" save_interval 5 > $O/e2e_lean_resume.log 2>&1; RC=$?
echo "lean_resume $(( $(date +%s) - T0 )) rc=$RC" >> $O/e2e_seconds.txt
rm -rf YOLOX_outputs/e2e_*/*.pth
# 5. N1: 2-GPU, uncached, 8 workers/GPU
for MODE in base cv2; do
  P="$V"; [ $MODE = cv2 ] && P="$V,worker_cv2_threads"
  python3 -m yolox.tools.train -f $EXP -expn n1_$MODE -d 2 -b 32 --fp16 -c /workspace/yolox_m.pth \
    --logger tensorboard seed 0 "${E1[@]}" max_epoch 40 no_aug_epochs 0 warmup_epochs 1 eval_interval 1000 data_num_workers 8 perf "$P" > $O/n1_$MODE.log 2>&1 &
  TP=$!
  sleep 75
  top -b -n 3 -d 5 -c > $O/n1_${MODE}_top.txt
  for pid in $(pgrep -f "yolox.tools.train|multiprocessing|pt_data_worker" ); do echo "$pid $(grep Threads /proc/$pid/status 2>/dev/null | awk '{print $2}') $(cat /proc/$pid/comm 2>/dev/null)"; done > $O/n1_${MODE}_threads.txt
  ps -eo pid,ppid,nlwp,pcpu,comm --sort=-pcpu | head -40 > $O/n1_${MODE}_ps.txt
  wait $TP
  cp YOLOX_outputs/n1_$MODE/train_log.txt $O/n1_${MODE}_train_log.txt 2>/dev/null
  rm -rf YOLOX_outputs/n1_$MODE/*.pth
done
echo "P5 DONE $(date -u +%FT%TZ)" >> /workspace/finish.log
