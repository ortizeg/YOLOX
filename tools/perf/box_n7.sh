#!/usr/bin/env bash
# GPU check of N7 (cuda_graphs) on a 2-GPU box with the basketball data (foundation-finetune models/yolox/setup_box.sh).
# Writes /workspace/n7 and "N7 DONE" to /workspace/finish.log.
#   1. tests/perf on the GPU
#   2. 60-step equivalence (bench_step --check, 1 GPU): reference = V + N6, then + cuda_graphs
#   3. 2-GPU step time, V+N6 vs V+N6+cuda_graphs, 300 iterations each, both orders: E1 recipe (800 +-5x32, 16/GPU)
#      and a lighter step (640 fixed, 8/GPU) where launch overhead is a larger share
#   4. fixed-800 profile pair; 5. 20-epoch basketball accuracy, seeds 0-2, one variant per GPU
set -x
cd /workspace/YOLOX; export PYTHONPATH=/workspace/YOLOX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=/workspace/n7; mkdir -p $O
EXP=exps/default/yolox_m_basketball_merged5_adamw.py
NG=$(nvidia-smi -L | wc -l)
V=nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map,ddp_bucket_view,async_ckpt,eval_persistent,eval_amp,cv_flip,preproc_nopad,hwc_transport,int_indices,fast_anno,simota_nosync,resize_nosync,anno_cache,head_nosync,worker_cv2_threads,host_label_counts,lean_ckpt,fused_optim_amp,iou_nosync,dense_loss,log_onesync
G=$V,cuda_graphs
E1=(input_size 800,800 test_size 640,640 multiscale_range 5 basic_lr_per_img 2.5e-05 mixup_prob 0.5)
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nproc --all
  python3 -c "import torch,cv2;print(torch.__version__,torch.version.cuda,cv2.__version__)"; echo "V=$V"; } > $O/env.txt 2>&1

[ -n "${SKIP_CHECKS:-}" ] || python3 -m pytest tests/perf -q -p no:cacheprovider -rs > $O/pytest_perf.txt 2>&1
[ -n "${SKIP_CHECKS:-}" ] || python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --check --warmup 0 --iters 60 --ref "$V" \
  --out $O/check.json --sets "$G" -- "${E1[@]}" > $O/check.log 2>&1
[ -n "${SKIP_CHECKS:-}" ] || python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --warmup 110 --iters 220 \
  --out $O/step_1gpu.json --sets "$V" "$G" "$V" "$G" -- "${E1[@]}" > $O/step_1gpu.log 2>&1

train() {  # name perf iters bpg "<extra>" ENV=VAL...
  local name=$1 p=$2 iters=$3 bpg=$4 extra=$5; shift 5
  env YOLOX_AUDIT_ITERS=$iters YOLOX_AUDIT_SPAN_EPOCHS=1 YOLOX_AUDIT_OUT=$O/mem_$name "$@" timeout -k 60 1800 \
    python3 -m yolox.tools.train -f $EXP -expn n7_$name -d $NG -b $(( bpg * NG )) --fp16 -c /workspace/yolox_m.pth \
    --logger tensorboard seed 0 "${E1[@]}" max_epoch 300 no_aug_epochs 0 warmup_epochs 1 eval_interval 1000 \
    data_num_workers 8 print_interval 10 $extra perf "$p" > $O/$name.log 2>&1
  echo "$name rc=$?" >> $O/runs.txt
  sleep 5; pkill -9 -f "[y]olox.tools.train"; pkill -9 -f "[m]ultiprocessing.spawn"; sleep 10
  cp YOLOX_outputs/n7_$name/train_log.txt $O/train_$name.txt 2>/dev/null; rm -rf YOLOX_outputs/n7_$name
}
for R in 1 2; do
  if [ $R = 1 ]; then ORD="V G"; else ORD="G V"; fi
  for S in $ORD; do P=$V; [ $S = G ] && P=$G
    train e1_${S}_r$R "$P" 300 16 ""
    train light_${S}_r$R "$P" 400 8 "input_size 640,640 multiscale_range 0"
  done
done
for S in V G; do P=$V; [ $S = G ] && P=$G
  train prof_$S "$P" 60 16 "input_size 800,800 multiscale_range 0" YOLOX_PROFILE=$O/prof_$S
done

# 5. accuracy
acc() { local GPU=$1 name=$2 p=$3
  for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPU python3 -m yolox.tools.train -f $EXP -expn acc_${name}_s$s -d 1 -b 16 --fp16 \
      -c /workspace/yolox_m.pth --cache ram --logger tensorboard seed $s "${E1[@]}" max_epoch 20 no_aug_epochs 3 \
      warmup_epochs 2 eval_interval 1 save_interval 5 perf "$p" > $O/acc_${name}_s$s.log 2>&1
    grep -h "best AP" YOLOX_outputs/acc_${name}_s$s/train_log.txt | tail -1 | sed "s/^/${name} s$s /" >> $O/ap.txt
    grep -h "Average Precision  (AP) @\[ IoU=0.50:0.95 | area=   all" YOLOX_outputs/acc_${name}_s$s/train_log.txt | tail -1 | sed "s/^/${name} s$s last /" >> $O/ap.txt
    cp YOLOX_outputs/acc_${name}_s$s/train_log.txt $O/acc_train_${name}_s$s.txt; rm -rf YOLOX_outputs/acc_${name}_s$s
  done; }
acc 0 V "$V" & acc 1 G "$G" & wait

python3 - > $O/digest.txt 2>&1 <<'PY'
import glob, json, os, re
O = "/workspace/n7"
for f in sorted(glob.glob(f"{O}/train_*_r*.txt")):
    t = [float(x) for x in re.findall(r"iter_time: ([0-9.]+)s", open(f).read())][10:]
    print(f"{os.path.basename(f)}: mean iter_time {sum(t)/max(len(t),1)*1000:.1f} ms over {len(t)} log lines")
for f in sorted(glob.glob(f"{O}/*.log")):
    for line in open(f):
        if "captured" in line and "cuda_graphs" in line:
            print(os.path.basename(f), line.split("perf cuda_graphs:")[1].strip()); break
for n in ["V", "G"]:
    f = f"{O}/prof_{n}/summary.json"
    if os.path.exists(f):
        p = json.load(open(f))
        print(f"prof {n}: step {p['step_wall_ms']:.1f} ms, GPU busy {p['gpu_busy_fraction']:.1%}, "
              f"{p['kernels_per_step']:.0f} kernels, host sync {sum(v['ms_per_step'] for v in p['host_sync'].values()):.1f} ms")
        for k, v in p["stages"].items():
            print(f"   {k:16s} cpu {v['cpu_ms']:7.2f}  cuda {v['cuda_ms']:7.2f}")
PY
echo "N7 DONE $(date -u +%FT%TZ)" >> /workspace/finish.log
