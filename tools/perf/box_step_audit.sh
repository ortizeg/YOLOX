#!/usr/bin/env bash
# On the 8x L40S box (Objects365 present, /workspace/YOLOX = this branch): audit the 8-GPU training step to choose
# the next speed fix (CUDA graphs, which need the remaining host syncs gone, vs an exact sparse SimOTA).
#   (a) memory: batch 16/GPU under a 24 GiB per-process cap (YOLOX_MEM_CAP_GB), 80 iterations of the real recipe,
#       plus 40 iterations pinned at the recipe's largest multiscale size (800) -> OOM or peak per rank
#   (b) host-sync audit on rank 0 (YOLOX_SYNC_AUDIT), 80 iterations: syncs per iteration by call site
#   (c) torch.profiler on rank 0 (YOLOX_PROFILE), 60 iterations: per-stage CPU/CUDA ms, GPU busy fraction,
#       with (d) nvidia-smi dmon on all GPUs alongside
# Every run stops after YOLOX_AUDIT_ITERS iterations (no checkpoint, no eval) under a throwaway -expn audit_*; the real
# run's YOLOX_outputs/<run> folder is never written. Writes /workspace/audit and "AUDIT DONE" to /workspace/finish.log.
#   bash tools/perf/box_step_audit.sh        [EXP=.. BPG=16 PERF=<flags> INIT=<ckpt, weights only> CAP=24]
set -x
cd /workspace/YOLOX; export PYTHONPATH=/workspace/YOLOX
# one math-library thread per process, as tools/perf/o365_train.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=/workspace/audit; mkdir -p $O
EXP=${EXP:-exps/perf/yolox_m_objects365_pretrain.py}
BPG=${BPG:-16}; CAP=${CAP:-24}; TMO=${TMO:-1800}
# gated set of the live Objects365 run (tools/perf/o365_supervise.sh, includes decode_reuse) + the five pass-5 flags
V=nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map,ddp_bucket_view
F=$V,async_ckpt,eval_persistent,eval_amp,cv_flip,preproc_nopad,hwc_transport,int_indices,fast_anno,simota_nosync,resize_nosync,anno_cache,decode_reuse
PERF=${PERF:-$F,head_nosync,worker_cv2_threads,host_label_counts,lean_ckpt,fused_optim_amp}
# optional weights-only init (steady-state assignments instead of a random-init model); read, never written
INIT=${INIT:-}; C=(); [ -n "$INIT" ] && [ -s "$INIT" ] && C=(-c "$INIT")

if pgrep -f "[y]olox.tools.train" > /dev/null; then
  echo "AUDIT ABORTED: a yolox.tools.train process is already running $(date -u +%FT%TZ)" >> /workspace/finish.log; exit 1
fi
NG=$(nvidia-smi -L | wc -l)
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nproc; free -g | head -2
  python3 -c "import torch,cv2;print(torch.__version__,torch.version.cuda,cv2.__version__)"
  git rev-parse HEAD 2>/dev/null; echo "exp $EXP bpg $BPG gpus $NG cap ${CAP}GiB init ${INIT:-none}"; echo "perf $PERF"; } > $O/env.txt 2>&1
echo "audit start $(date -u +%FT%TZ)" >> /workspace/finish.log

run() {  # name iters "<extra exp opts>" ENV=VAL...
  local name=$1 iters=$2 extra=$3; shift 3
  local T0=$(date +%s)
  env YOLOX_AUDIT_ITERS=$iters YOLOX_AUDIT_OUT=$O/mem_$name "$@" \
    timeout -k 60 $TMO python3 -m yolox.tools.train -f $EXP -expn audit_$name -d $NG -b $(( BPG * NG )) --fp16 \
    "${C[@]}" --logger tensorboard data_num_workers 8 eval_interval 1000 $extra perf "$PERF" > $O/$name.log 2>&1 &
  local PID=$!
  # an OOM on one rank leaves the others waiting in NCCL (30 min watchdog): stop the run 60 s after the first OOM
  while kill -0 $PID 2>/dev/null; do
    if grep -qi "out of memory" $O/$name.log; then sleep 60; kill $PID 2>/dev/null; break; fi; sleep 5
  done
  wait $PID; local RC=$?
  # leftovers (spawned ranks, loader workers) of this run; the start-up check guarantees no other training here
  sleep 5; pkill -9 -f "[y]olox.tools.train" 2>/dev/null; pkill -9 -f "[m]ultiprocessing.spawn" 2>/dev/null
  for _ in $(seq 24); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
  local OOM=$(grep -ci "out of memory" $O/$name.log)
  echo "$name rc=$RC wall_s=$(( $(date +%s) - T0 )) oom_lines=$OOM iters=$iters extra='$extra'" >> $O/runs.txt
  cp YOLOX_outputs/audit_$name/train_log.txt $O/train_$name.txt 2>/dev/null
  rm -rf YOLOX_outputs/audit_$name
}

# (a) memory under a 24 GiB cap: the recipe as is, then pinned at 800 px (multiscale max of 640 +- 5x32)
run mem 80 "" YOLOX_MEM_CAP_GB=$CAP
run mem800 40 "input_size 800,800 multiscale_range 0" YOLOX_MEM_CAP_GB=$CAP
# (b) host syncs on rank 0: warm-up 30 iterations, then 20 audited ones
run sync 80 "" YOLOX_SYNC_AUDIT=$O/sync_audit.json
# (c) profile on rank 0, schedule(wait=30, warmup=5, active=10), with (d) dmon on every GPU
nvidia-smi dmon -s pucm -d 1 -o T > $O/dmon_profile.txt 2>&1 & DM=$!
run profile 60 "" YOLOX_PROFILE=$O/profile
kill $DM 2>/dev/null

# one-page digest (peak memory per rank, top sync sites, profile headline)
python3 - > $O/digest.txt 2>&1 <<'PY'
import glob, json, os, sys
O = "/workspace/audit"
out = sys.stdout.write
for d in sorted(glob.glob(f"{O}/mem_*")):
    for f in sorted(glob.glob(f"{d}/mem_rank*.json")):
        r = json.load(open(f))
        out(f"{os.path.basename(d)} rank {r['rank']}: oom={r['oom']} iters={r['iters_done']} "
            f"peak alloc {r.get('max_allocated_gib', 0):.2f} GiB reserved {r.get('max_reserved_gib', 0):.2f} GiB "
            f"cap {r['mem_cap_gb']} {r.get('error') or ''}\n")
if os.path.exists(f"{O}/sync_audit.json"):
    s = json.load(open(f"{O}/sync_audit.json"))
    out(f"\nsyncs/iter {s['total_syncs_per_iter']:.2f} over {s['meta']['iters_audited']} iters\n")
    for k, v in list(s["sites"].items())[:25]:
        out(f"  {v['count_per_iter']:6.2f}/iter  [{v['stage']}]  {k}\n")
if os.path.exists(f"{O}/profile/summary.json"):
    p = json.load(open(f"{O}/profile/summary.json"))
    out(f"\nstep {p['step_wall_ms']:.1f} ms, GPU busy {p['gpu_busy_fraction']:.1%} "
        f"(excl NCCL {p['gpu_busy_fraction_excl_nccl']:.1%}), {p['kernels_per_step']:.0f} kernels/step, "
        f"host sync {sum(v['ms_per_step'] for v in p['host_sync'].values()):.1f} ms/step\n")
    for k, v in p["stages"].items():
        out(f"  {k:24s} cpu {v['cpu_ms']:7.2f} ms  cuda {v['cuda_ms']:7.2f} ms\n")
PY
echo "AUDIT DONE $(date -u +%FT%TZ)" >> /workspace/finish.log
