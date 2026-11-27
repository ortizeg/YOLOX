#!/usr/bin/env bash
# GPU check of the N6 flags (iou_nosync, dense_loss, log_onesync) on a 2-GPU box with the basketball data
# (foundation-finetune models/yolox/setup_box.sh). Writes /workspace/n6 and "N6 DONE" to /workspace/finish.log.
#   1. tests/perf on the GPU
#   2. 60-step equivalence (bench_step --check, 1 GPU): reference = the gated set below, then + each flag and all three
#   3. host-sync audit of a real 2-GPU training run (rank 0, 20 iterations after 30 warm-up) without and with N6
#   4. 2-GPU training step time without and with N6, 300 iterations each, both orders, plus a fixed-size profile pair
set -x
cd /workspace/YOLOX; export PYTHONPATH=/workspace/YOLOX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=/workspace/n6; mkdir -p $O
EXP=exps/default/yolox_m_basketball_merged5_adamw.py
NG=$(nvidia-smi -L | wc -l)
# the set the 8x L40S Objects365 run and step audit used (tools/perf/box_step_audit.sh), minus the tier-C decode_reuse
V=nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map,ddp_bucket_view,async_ckpt,eval_persistent,eval_amp,cv_flip,preproc_nopad,hwc_transport,int_indices,fast_anno,simota_nosync,resize_nosync,anno_cache,head_nosync,worker_cv2_threads,host_label_counts,lean_ckpt,fused_optim_amp
N6=$V,iou_nosync,dense_loss,log_onesync
E1=(input_size 800,800 test_size 640,640 multiscale_range 5 basic_lr_per_img 2.5e-05 mixup_prob 0.5)
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nproc --all
  python3 -c "import torch,cv2;print(torch.__version__,torch.version.cuda,cv2.__version__)"; echo "V=$V"; } > $O/env.txt 2>&1

# 1
python3 -m pytest tests/perf -q -p no:cacheprovider -rs > $O/pytest_perf.txt 2>&1

# 2
python3 tools/perf/bench_step.py -f $EXP -c /workspace/yolox_m.pth --check --warmup 0 --iters 60 --ref "$V" --out $O/check.json \
  --sets "$V,iou_nosync" "$V,dense_loss" "$V,log_onesync" "$N6" -- "${E1[@]}" > $O/check.log 2>&1

train() {  # name perf iters "<extra>" ENV=VAL...
  local name=$1 p=$2 iters=$3 extra=$4; shift 4
  env YOLOX_AUDIT_ITERS=$iters YOLOX_AUDIT_OUT=$O/mem_$name "$@" timeout -k 60 1800 python3 -m yolox.tools.train -f $EXP \
    -expn n6_$name -d $NG -b $(( 16 * NG )) --fp16 -c /workspace/yolox_m.pth --logger tensorboard seed 0 "${E1[@]}" \
    max_epoch 200 no_aug_epochs 0 warmup_epochs 1 eval_interval 1000 data_num_workers 8 print_interval 10 $extra \
    perf "$p" > $O/$name.log 2>&1
  echo "$name rc=$?" >> $O/runs.txt
  sleep 5; pkill -9 -f "[y]olox.tools.train"; pkill -9 -f "[m]ultiprocessing.spawn"; sleep 10
  cp YOLOX_outputs/n6_$name/train_log.txt $O/train_$name.txt 2>/dev/null; rm -rf YOLOX_outputs/n6_$name
}

# 3
train sync_V "$V" 80 "" YOLOX_SYNC_AUDIT=$O/sync_V.json
train sync_N6 "$N6" 80 "" YOLOX_SYNC_AUDIT=$O/sync_N6.json

# 4
for R in 1 2; do
  if [ $R = 1 ]; then A=V; B=N6; else A=N6; B=V; fi
  for S in $A $B; do P=$V; [ $S = N6 ] && P=$N6; train time_${S}_r$R "$P" 300 ""; done
done
for S in V N6; do P=$V; [ $S = N6 ] && P=$N6
  train prof_$S "$P" 60 "input_size 800,800 multiscale_range 0" YOLOX_PROFILE=$O/prof_$S
done

# digest: syncs per iteration by site, mean iter_time over iterations 100-300, profile headline
python3 - > $O/digest.txt 2>&1 <<'PY'
import glob, json, os, re
O = "/workspace/n6"
for n in ["V", "N6"]:
    f = f"{O}/sync_{n}.json"
    if os.path.exists(f):
        s = json.load(open(f))
        print(f"sync {n}: {s['total_syncs_per_iter']:.2f}/iter over {s['meta']['iters_audited']} iters")
        for k, v in list(s["sites"].items())[:8]:
            print(f"   {v['count_per_iter']:5.2f}  [{v['stage']}]  {k}")
for f in sorted(glob.glob(f"{O}/train_time_*.txt")):
    # iter resets every epoch (14 iterations at batch 32): drop the first 10 log lines (~100 iterations) instead
    t = [float(x) for x in re.findall(r"iter_time: ([0-9.]+)s", open(f).read())][10:]
    print(f"{os.path.basename(f)}: mean iter_time {sum(t)/max(len(t),1)*1000:.1f} ms over {len(t)} log lines")
for n in ["V", "N6"]:
    f = f"{O}/prof_{n}/summary.json"
    if os.path.exists(f):
        p = json.load(open(f))
        print(f"prof {n}: step {p['step_wall_ms']:.1f} ms, GPU busy {p['gpu_busy_fraction']:.1%}, "
              f"{p['kernels_per_step']:.0f} kernels, host sync {sum(v['ms_per_step'] for v in p['host_sync'].values()):.1f} ms")
        for k in ["forward", "backward", "head_targets", "head_loss"]:
            v = p["stages"].get(k)
            if v:
                print(f"   {k:14s} cpu {v['cpu_ms']:7.2f}  cuda {v['cuda_ms']:7.2f}")
PY
echo "N6 DONE $(date -u +%FT%TZ)" >> /workspace/finish.log
