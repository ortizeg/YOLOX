#!/usr/bin/env bash
# torch.compile benchmark on a fresh 1-GPU box (no dataset needed). Writes /workspace/compile/* and COMPILE DONE.
set -x
cd /workspace; O=/workspace/compile; mkdir -p $O
pip install -q loguru tabulate thop tqdm psutil tensorboard pycocotools opencv-python-headless "torch==2.5.1" "torchvision==0.20.1" 2>&1 | tail -1
mkdir -p YOLOX && tar -C YOLOX -xzf yolox_src.tgz && cd YOLOX && export PYTHONPATH=/workspace/YOLOX
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $O/gpu.txt; nproc >> $O/gpu.txt
python3 tools/perf/compile_bench.py --variants eager bb bb_head bb_maxat dyn --out $O/compile_t251.json > $O/t251.log 2>&1
# newer torch: dynamic-shape compile (one graph for all sizes) crashed in 2.5.1's inductor
python3 -m venv /workspace/v27 && /workspace/v27/bin/pip install -q "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -1
/workspace/v27/bin/pip install -q loguru tabulate thop tqdm psutil pycocotools opencv-python-headless numpy 2>&1 | tail -1
/workspace/v27/bin/python tools/perf/compile_bench.py --variants eager bb dyn --out $O/compile_t271.json > $O/t271.log 2>&1
echo "COMPILE DONE" >> /workspace/finish.log
