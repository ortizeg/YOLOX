#!/usr/bin/env bash
# 8x RTX 4090 throughput on the Objects365 sample (patch0+patch1), comparable to the 8x L40S perf_8g_w8 row.
set -x
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /workspace; O=/workspace/o365; mkdir -p $O
B=https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train
pip install -q loguru tabulate thop tqdm psutil tensorboard pycocotools opencv-python-headless "torch==2.5.1" "torchvision==0.20.1" 2>&1 | tail -1
(apt-get update -qq && apt-get install -y -qq aria2) > /dev/null 2>&1
mkdir -p YOLOX && tar -C YOLOX -xzf yolox_src.tgz
D=/workspace/YOLOX/datasets/objects365; mkdir -p $D/annotations $D/train
T0=$(date +%s)
for f in patch0.tar.gz patch1.tar.gz zhiyuan_objv2_train.tar.gz; do
  aria2c -q -x 16 -s 16 -k 16M --max-tries=5 --retry-wait=5 -d /workspace/dl -o $f "$B/$f" || echo "DOWNLOAD FAILED $f" >> /workspace/finish.log
done
echo "download_seconds $(( $(date +%s) - T0 ))" > $O/download.txt; du -sh /workspace/dl >> $O/download.txt
echo "DOWNLOAD DONE" >> /workspace/finish.log
tar -C $D/train -xzf /workspace/dl/patch0.tar.gz && tar -C $D/train -xzf /workspace/dl/patch1.tar.gz
tar -C /workspace/dl -xzf /workspace/dl/zhiyuan_objv2_train.tar.gz
cd /workspace/YOLOX && python3 - <<'PY' > /workspace/o365/subset.txt 2>&1
import json, os, glob
D = "/workspace/YOLOX/datasets/objects365"
have = {os.path.basename(p): os.path.relpath(p, f"{D}/train") for p in glob.glob(f"{D}/train/**/*.jpg", recursive=True)}
js = json.load(open(glob.glob("/workspace/dl/**/zhiyuan_objv2_train.json", recursive=True)[0]))
imgs = [dict(im, file_name=have[os.path.basename(im["file_name"])]) for im in js["images"] if os.path.basename(im["file_name"]) in have]
ids = {im["id"] for im in imgs}
anns = [a for a in js["annotations"] if a["image_id"] in ids]
json.dump({"images": imgs, "annotations": anns, "categories": js["categories"]}, open(f"{D}/annotations/train_subset.json", "w"))
v = {i["id"] for i in imgs[:500]}
json.dump({"images": imgs[:500], "annotations": [a for a in anns if a["image_id"] in v], "categories": js["categories"]},
          open(f"{D}/annotations/train_subset_val.json", "w"))
print("subset images", len(imgs), "annotations", len(anns), "boxes/img", round(len(anns) / max(len(imgs), 1), 1))
PY
echo "PREP DONE" >> /workspace/finish.log
EXP=exps/perf/yolox_m_objects365.py
V=nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map,ddp_bucket_view
P3=cv_flip,preproc_nopad,hwc_transport,int_indices,fast_anno,simota_nosync,resize_nosync,anno_cache
NG=$(nvidia-smi -L | wc -l)
run() {  # name workers perf
  local T0=$(date +%s)
  timeout 1500 python3 -m yolox.tools.train -f $EXP -expn sc_$1 -d $NG -b $((16 * NG)) --fp16 -o --logger tensorboard \
    max_epoch 1 no_aug_epochs 0 warmup_epochs 1 data_num_workers $2 print_interval 10 perf $3 > $O/run_$1.log 2>&1 &
  local PID=$!
  ( peak=0; while kill -0 $PID 2>/dev/null; do u=$(free -m | awk '/Mem/{print $3}'); [ $u -gt $peak ] && peak=$u && echo $peak > $O/peakmem_$1.txt; sleep 2; done ) &
  local snap=0
  while kill -0 $PID 2>/dev/null; do
    n=$(grep -c "iter: " $O/run_$1.log 2>/dev/null)
    [ "${n:-0}" -ge 1 ] && [ ! -f $O/firstiter_$1.txt ] && echo $(( $(date +%s) - T0 )) > $O/firstiter_$1.txt
    if [ "${n:-0}" -ge 15 ] && [ $snap = 0 ]; then snap=1
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv > $O/smi_$1.txt
      top -bn1 | head -5 >> $O/smi_$1.txt
    fi
    [ "${n:-0}" -ge 26 ] && break; sleep 5
  done
  sleep 2; pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null; sleep 5; pkill -f "sc_$1" 2>/dev/null; sleep 5
  echo "$1 wall $(( $(date +%s) - T0 )) peak_used_MiB $(cat $O/peakmem_$1.txt 2>/dev/null) first_iter_s $(cat $O/firstiter_$1.txt 2>/dev/null) oom $(grep -ci 'out of memory' $O/run_$1.log)" >> $O/runs.txt
  cp YOLOX_outputs/sc_$1/train_log.txt $O/train_$1.txt 2>/dev/null
}
run perf_8g_w8 8 "$V,$P3"
run reuse_8g_w8 8 "$V,$P3,decode_reuse"
run reuse_8g_w12 12 "$V,$P3,decode_reuse"
nvidia-smi -L > $O/gpus.txt; lscpu | grep -E "Model name|^CPU\(s\)" >> $O/gpus.txt
echo "BENCH DONE" >> /workspace/finish.log
