# YOLOX training speed-ups: ledger (2026-11-26)

Branch `perf/train-speed` (base `8d71c7f` = published training code `64c55e2` + merged5 exp). Every change is an
opt-in flag (`yolox/utils/perf.py`); with `perf ""` (the default) every code path is the original YOLOX one.
Enable with the exp opt `perf <comma list>` or `perf all`.

**Where the evidence lives.**
- **This repo:**
  - code and flags in `yolox/utils/perf.py`
  - equivalence tests in `tests/perf/`
  - benchmark and profiling tools in `tools/perf/`
  - raw benchmark outputs and training logs in `docs/perf/results/`
  - Objects365/COCO configs and scripts in `exps/perf/` and `tools/perf/o365_*.sh`
- **ortizeg/foundation-finetune:**
  - 3-seed accuracy runs: `results/yolox/{perf,q95,norot,ema998,ema999,gref,gfull,greuse,gb32}_e1_s*`
  - bootstrap outputs: `results/yolox/bootstrap_*.json`, made by `models/rtmdet/bootstrap.py`
  - launch scripts as run: `vast/yolox_perf/`
  - Objects365 run log: `results/yolox/o365_m_e20/`
- Sections are in the order the work happened. Later sections supersede "not done yet" notes in earlier ones.

## Claim

On the tuned YOLOX-M basketball recipe (E1: train 800 ±5×32, batch 16, fp16, AdamW, EMA), one RTX 4090:

| measure | original | with speed-ups | speed-up |
|---|---:|---:|---:|
| GPU step, ms/iter (fixed batches, multiscale) | 279 | 118 | **2.36×** |
| real training, median iter_time | 0.310 s | 0.118 s | **2.63×** |
| real training, epochs 1–20 incl. eval | 235 s | 115 s | **2.04×** |
| real training, 20-epoch wall clock incl. start-up | 266 s | 138 s | **1.93×** |
| data path, 8 workers (img/s) | 106 | 128 | 1.21× |

Accuracy is unchanged (3 seeds each, val selects the checkpoint, test predicted once, harness scorer):

| | val mAP@50:95 | test mAP@50:95 | test mAP@75 |
|---|---:|---:|---:|
| original E1 (`yolox_e1_s0-2`, published) | 0.6663 ± 0.0060 | 0.6920 ± 0.0075 | 0.805 |
| speed-ups (`perf_e1_s0-2`) | 0.6755 ± 0.0021 | 0.6890 ± 0.0051 | 0.805 |

Clip-clustered paired bootstrap, test: perf − original **−0.3 pt [−1.2, +0.4]**, ahead on 1 of 3 clips: a tie
(foundation-finetune `results/yolox/bootstrap_perf_vs_e1.json`). The flag set used for every number above (the "validated set"):
`nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,channels_last,uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map`.

## Equivalence tests

- **CPU, pytest** (`tests/perf`, 22 tests; plus the existing 229): data pipeline bit-identical over 48 mosaic /
  non-mosaic samples at 640 and 800 and each data flag alone, including the tiny-box fallback path; batched SimOTA
  bit-identical losses and gradients (5 and 80 classes, empty images, crowded duplicate boxes, L1 on); class-cost
  algebra allclose; SPP chain identical; Focus conv allclose (forward and gradients); foreach EMA identical.
  251 passed on macOS (pixi) and on the box (torch 2.5.1+cu124).
- **GPU, 60 training iterations, deterministic algorithms** (`results/check2.json`): baseline reproduces itself
  exactly; `no_empty_cache`, `ema_foreach` and all tier-A flags together (`tierA`, data flags included) give
  **identical losses and final EMA weights**. `simota_batch` and `sppf` are not bit-identical over 60 steps
  (max loss difference 0.8% and 8.6%; trajectories diverge once a step differs), so both are tier B.
- **GPU, assignment agreement** (`results/assign.json`): batched SimOTA vs the original loop on 16 real batches
  from the same forward pass: **16/16 batches with identical targets** (about 2,200–3,000 positives each). The
  60-step drift therefore comes from rare later events, not systematic assignment differences; judged by the
  3-seed accuracy runs above.

| flag | tier | GPU result |
|---|---|---|
| no_empty_cache, ema_foreach, ddp_bucket_view, uint8_transport, skip_noop_resize, hsv_lut, int_mixup, class_map | A | bit-identical |
| simota_batch | B | identical targets 16/16 batches; 60-step drift ≤ 0.8% |
| sppf | B | forward exact; max-pool backward ties route differently in fp16 |
| cls_cost_algebra, fused_optim, channels_last, focus_conv, nondeterministic, compile | B | tolerance-equal |

## Where the time went (GPU step, RTX 4090, `results/run3_step.log`)

| cumulative flags | ms/iter | speed-up |
|---|---:|---:|
| original (cuDNN deterministic, as every seeded run) | 279.2 | 1.00× |
| + nondeterministic | 289.7 | 0.96× (no gain) |
| + no_empty_cache | 238.5 | 1.17× |
| + simota_batch | 166.9 | 1.67× |
| + cls_cost_algebra | 165.9 | 1.68× |
| + ema_foreach | 148.0 | 1.89× |
| + fused_optim | 144.9 | 1.93× |
| + sppf, focus_conv | 144.8 | 1.93× (no gain) |
| + channels_last | 118.4 | **2.36×** |
| + compile (not in the validated set) | 114.8 | 2.43× |

Single flags: no_empty_cache 241.3 (1.16×), simota_batch 170.9 (1.63×), channels_last 254.7 (1.10×), compile
256.5 (1.09×). Batched SimOTA is the largest single win: the original runs SimOTA per image with several
host-device syncs and a `torch.cuda.empty_cache()` each time.

## Dropped or no measurable gain

- **compile**: `torch.compile` of backbone + neck recompiles for each of the 11 multiscale sizes (about 5 min
  each on the 4090), so a 20-epoch run took > 40 min; after warm-up an epoch is 4 s vs 12 s, but the steady-state
  gain over channels_last is 3%. Not worth it for runs of this length; dynamic-shape compile crashed in torch
  2.5.1's inductor.
- **nondeterministic, sppf, focus_conv**: no measurable step gain on this GPU and model (kept in the validated set
  because they're harmless, but they shouldn't be credited).
- **cls_cost_algebra**: ~0 at 5 classes as expected; it removes the `[gt × anchor × class]` tensor, so its value
  is at 80 / 365 classes (COCO / Objects365), not measured here.

## Not done yet (as of pass 1; see later sections)

- Since done: multi-GPU and Objects365 scale (8× L40S section), the first tier-C item (`decode_reuse`, pass 4), and
  bigger batch (failed the pass-4 gate). Still open:
- Tier-C items (combined mosaic warp, GPU augmentation, progressive resizing, bigger batch) and D6 (turbojpeg):
  they change results and need their own accuracy runs.
- Multi-GPU (DDP) and Objects365-scale data: untested. The data path ran at 106 → 128 img/s with 8 workers on
  uncached JPEGs, so at 8 fast GPUs the disk cache (`--cache disk`) and more workers per GPU will matter.

## Cost

About $1.70 of vast credit: one RTX 4090 bench box (tests, GPU checks, ablation, data, 2× end-to-end pairs) and
three RTX 4090 accuracy boxes (full 200-epoch E1, 843 / 1352 / 1170 s training each). All destroyed.

## Reproduce

```
pixi run -e dev test                                   # CPU equivalence tests
tools/perf/launch_bench.sh <label> "<vast query>"     # box: pytest, GPU check, ablation, data, e2e
python tools/perf/assign_agreement.py -f <exp> -c yolox_m.pth --out assign.json -- <exp opts>
```

## Pass 2 (2026-11-26, fresh RTX 4090; `results/pass2/`)

**Profile** (ms per step, CUDA events, 110 iterations over all 11 multiscale sizes):

| stage | original | validated set |
|---|---:|---:|
| backbone + neck forward | 33.5 | 29.5 |
| head forward + SimOTA + loss | 113.4 | **19.0** |
| backward | 90.1 | 69.7 |
| optimizer | 4.6 | 2.5 |
| EMA | 19.8 | 1.1 |
| total | 261.4 | 121.7 |

The step is now compute-bound: conv backward 33%, conv forward 15%, batch-norm about 16%, SiLU about 10% of CUDA
time (`profile.log`). Remaining GPU gains need kernel fusion (BN + SiLU; compile was dropped for recompiles) or
model changes.

**End to end, 20 epochs, same box:** original 212 s, validated set 128 s (1.66×; this box's start-up and eval share
were larger than pass 1's), validated + `async_ckpt,eval_persistent,eval_amp` 134 s: no measurable gain.

**Correction to the pass-2 premise.** YOLOX logs iter_time only at iterations 10 and 20 of 29, so summing logged
iter_time undercounted iterations by about a third. Corrected, the 128 s run is about 78 s iterations, 14 s eval
(4 evaluations), 10 s first-epoch cuDNN autotuning for 11 sizes (one-time), 8 s checkpoints (0.38 s per 290 MiB
write on this disk) and 7 s start-up. Outside-iteration time is about 35%, much of it one-time, not the "40% per
epoch" estimated before pass 2.

**Eval check** (`eval_check.json`, same checkpoint, 3 evaluations each): original 2.80 / 1.11 / 1.15 s, persistent
workers 1.15 / 1.04 / 0.85 s, + fp16 autocast 1.45 / 0.98 / 0.81 s. AP50:95 0.6164 → 0.6163 and AP50 0.9372 → 0.9366
with fp16: small but not identical, so `eval_amp` stays tier B and is not recommended for checkpoint selection.

**Disk cache:** the lazy write-through cache (`--cache lazy`) is implemented and bit-identical (test). Cache-format
benchmark: `tools/perf/bench_cache.py`. For Objects365, whose images are mostly about 640 px already, a re-encoded
copy saves little (108 vs 116 KB per image), so reading originals is the plan.

## Pass 3 on GPU (2026-11-26, same RTX 4090 as pass 2; `results/pass3/`)

- GPU step: validated set 117.8 ms → + `simota_nosync` 116.6 ms (1%; the step is compute-bound).
- Data loader (uncached JPEGs, 800): 8 workers 105.2 (original) → 126.6 (pass-1 data flags) → **144.7 img/s** with
  `cv_flip,preproc_nopad,hwc_transport,int_indices` (+14% over pass 1, +38% over the original); 4 workers 52.8 → 61.2 → 68.7.
- End to end, 20 epochs, back to back: pass-1/2 flags 134 s → + all pass-3 flags **115 s** (1.17×). The median iter_time
  is unchanged (0.122 vs 0.124 s), so the gain is outside the logged iterations; it's a single pair, and pass 2 showed
  about ±5% run-to-run noise. Against the original on this box (212 s): **1.84×**. Final val AP 0.669 vs 0.673.
- Box pytest: 282 passed, **2 failed**: `test_pass3_pipeline_identical[size*-True]`, where `img_info` from the mosaic path
  differs under `hwc_transport` ((640, 3) original vs (640, 640)). It passes on macOS with the same code. `img_info`
  is unused in training, but the tier-A claim for `hwc_transport` is on hold until this is explained.

## 8-GPU Objects365 scaling (2026-11-26, 8× L40S, 128 CPUs, 1 TB RAM; `results/o365_scale/`)

Objects365 v2 sample: patch0 + patch1, 69,519 images, 1.36 M boxes (19.5 per image), 365 classes, 640 input,
YOLOX-M with YOLOX's own recipe (exps/perf/yolox_m_objects365.py), batch 16 per GPU, uncached JPEGs, about 260
iterations per run. img/s = batch / median logged iter_time; data_time stayed at 0.001–0.002 s in every run.

| run | img/s | scaling eff. | start-up | peak RAM |
|---|---:|---:|---:|---:|
| original, 1 GPU | 82.7 | | 48 s | 17 GB |
| speed-ups, 1 GPU | 146.8 (1.78×) | | 19 s | 16 GB |
| original, 8 GPU, 4 workers/GPU | 455.5 | 69% | 211 s | 124 GB |
| **speed-ups, 8 GPU, 8 workers/GPU** | **1,142.9 (2.51×)** | **97%** | 46 s | 56 GB |
| speed-ups, 8 GPU, 4 workers/GPU | 1,113.0 | 95% | 35 s | 43 GB |
| speed-ups, 8 GPU, 12 workers/GPU | 948.1 | | 59 s | 68 GB |
| speed-ups without anno_cache, 8 GPU, 8 workers | 825.8 | | 148 s | 159 GB |
| speed-ups + bf16 (instead of fp16), 8 GPU, 8 workers | 984.6 | | 46 s | 54 GB |

- Not data-bound at 4–8 workers per GPU; 12 is slower (CPU contention). Use 4–8.
- bf16 is 14% slower than fp16 on this Ada GPU: keep fp16.
- anno_cache: 3× less RAM, start-up 148 → 46 s, and faster iterations; at full Objects365 scale (≈ 19× these
  annotations) the original path would need far more RAM per rank.
- Projected Objects365 v2 cost at 1,143 img/s on 8× L40S ($9.60/h), iterations only: 25 min per epoch, so 10 epochs
  ≈ 4.2 h ≈ $41 and 50 epochs (≈ YOLO26's image passes) ≈ 21 h ≈ $203; COCO 80 epochs ≈ 2.3 h ≈ $22. Add about 10%
  for evaluation and checkpoints.

## torch.compile (2026-11-26, RTX 4090, YOLOX-M, batch 16, synthetic full training step, 11 multiscale sizes 640–960)

Raw: `docs/perf/results/compile/compile_t251.json`, `compile_t271.json` (`tools/perf/compile_bench.py`).

| torch | variant | total compile | mean step | vs 2.5.1 eager |
|---|---|---:|---:|---:|
| 2.5.1+cu124 (cuDNN 9.1) | eager | 0.5 min | 113.3 ms | 1.00× |
| 2.5.1 | backbone+neck, static per size | 16.0 min | 108.9 ms | 1.04× |
| 2.5.1 | + head towers | 5.2 min | 113.8 ms | 1.00× |
| 2.5.1 | backbone, max-autotune | 49.6 min | 107.8 ms | 1.05× |
| 2.5.1 | backbone, dynamic=True (one graph) | 5.6 min once | 110.8 ms | 1.02× |
| 2.7.1+cu126 (cuDNN 9.5) | eager | 0.8 min | 113.8 ms | 1.00× |
| 2.7.1 | backbone+neck, static per size | 13.8 min | 109.4 ms | 1.04× |
| 2.7.1 | backbone, dynamic=True (one graph) | 4.4 min once | 109.3 ms | 1.04× |

Conclusion: the step is cuDNN conv/BN-bound, so Inductor fusion removes only ~4%; a torch/cuDNN upgrade alone gives nothing.
Best option: 2.7.1 + dynamic, 4.4 min once for 4% → ~50 min off a 21 h Objects365 run, break-even ≈ 1.8 h of training (a
20-epoch basketball run never breaks even). Not a default; candidate for the Objects365 run only.

## Pass 4: tier-C items (in progress)

- `decode_reuse` (C): mosaic/mixup partners come from a per-worker ring buffer of recent fresh decodes (exp
  `decode_reuse_pool=64`, `decode_reuse_prob=1.0`); the primary image is always freshly decoded, and joins the pool after
  its own sample. JPEG decodes per sample 5 → 1 (≈ 1 + 4·(1−prob)). Tier-C flags are excluded from `perf all`. Tests:
  `tests/perf/test_perf_pass4.py` (flag off = original draws; ≤ 1.2 decodes/sample at prob 1; pool never mutated).
- Bigger batch per GPU: no code; measure with `bench_step.py --batch 16 24 32 48` and 3-seed accuracy.

### Pass-4 accuracy gate (2026-11-27, GCP A100 40 GB spot, tuned E1 recipe, 3 seeds, clip-clustered bootstrap)

All arms on the same hardware. ref = pass-1 validated set; full = ref + pass-2/3 flags. Raw: foundation-finetune
`results/yolox/g{ref,full,reuse,b32}_e1_s{0,1,2}`, `bootstrap_g{full_vs_gref,reuse,b32}.json`.

| arm | test mAP (s0/s1/s2) | mean | val | Δ (CI) | verdict |
|---|---|---:|---:|---|---|
| ref | 0.682 / 0.684 / 0.683 | 0.683 | 0.675 | — | baseline |
| full (all pass-2/3 flags) | 0.681 / 0.691 / 0.689 | 0.687 | 0.674 | +0.41 vs ref [−0.02, +0.96], 2/3 clips | **pass** |
| full + decode_reuse | 0.669 / 0.697 / 0.689 | 0.685 | 0.671 | −0.17 vs full [−1.03, +0.43]; +0.23 vs ref [−0.09, +0.82] | **pass** (wider seed spread) |
| full @ batch 32 | 0.677 / 0.680 / 0.682 | 0.680 | 0.671 | −0.73 vs full [−2.01, −0.10], 0/3 clips | **fail**, dropped |

Caveat on batch 32: basketball has 464 train images, so batch 32 halves the iterations (14/epoch); on Objects365 that
effect is absent, but the rule is the rule and the Objects365 run uses batch 16 per GPU.

### 8× H200 throughput on full Objects365 (1.74 M images; mean over logged iterations after 50; raw `results/h200_bench/`)

| batch/GPU | workers/GPU | decode_reuse | s/iter | img/s |
|---:|---:|:-:|---:|---:|
| 16 | 12 | | 0.111 | 1,151 |
| 16 | 12 | ✓ | 0.102 | 1,258 |
| 32 | 12 | | 0.173 | 1,476 |
| 32 | 16 | | 0.171 | 1,495 |
| 32 | 12 | ✓ | 0.134 | 1,910 |
| 64 | 16 | ✓ | 0.222 | 2,309 |

At batch 16 the H200 node is no faster than 8× L40S (1,143 img/s): the step is overhead/loader-bound, not GPU-bound.
Bug found and fixed on the way: the rank that builds `anno_cache` kept the parsed COCO object (28.6 GB at Objects365
scale) and spawned workers each received a pickled copy, stalling start-up > 12 min; the builder now re-inits from the
cache it just wrote (`coco.py`).

## Decision: Objects365 hardware and when to re-pretrain (2026-11-27)

**Hardware.** Use 8× L40S for Objects365 pretraining, not 8× H200. At batch 16 per GPU both run at about the same speed
(1,143 vs 1,151 img/s without `decode_reuse`), and the L40S node costs about a quarter as much per hour ($9.60 vs $38.12):
about $4 vs $12 per Objects365 epoch, about $80 vs $240 for 20 epochs. The H200 is only worth it at batch 64 per GPU
(2,309 img/s, about $8 per epoch), which hasn't passed an accuracy check. Before the next run, check on the L40S box:
throughput on the full 1.74 M images with `decode_reuse`; at least 64 CPU cores (8 workers per GPU); and a 15-min
8× RTX 4090 comparison, which may be cheaper still.

**When to re-pretrain on Objects365.**

| Change | Objects365 again? | How to test it |
|---|---|---|
| Backbone or neck architecture (blocks, widths, normalisation, activation, input stem) | yes | full Objects365 → COCO |
| Head architecture (new head, different output parameterisation such as DFL bins) | no | COCO fine-tune from the Objects365 checkpoint; re-initialise the head in the baseline arm too |
| Loss, assigner, label assignment, augmentation, schedule | no | COCO fine-tune from the Objects365 checkpoint |

Rules for the COCO screening runs:
- Compare against a baseline arm fine-tuned from the same checkpoint with the same recipe and seeds, not against
  the official 46.9.
- Use at least 2 seeds when the expected gain is under about 0.5 mAP.
- A gain is good evidence. A null result is not conclusive: features pretrained under the old loss can hide a benefit
  that would show from scratch.
- Re-pretrain once with the final loss before a release, if the COCO gain is near the noise.
- The final decision is on basketball test (3 seeds, clip-clustered CI). COCO is the intermediate gate.

## Why the H200 GPUs sit at 35–40% (2026-11-27, live Objects365 run, batch 16/GPU)

- **Not data-starved.** data_time is 0.001 s on every logged step, and 68% of the 192 CPU cores are idle. YOLOX's
  data_time includes the wait for the next batch from the loader, so the GPU never waits on data.
- **Per-step overhead.** Each ~80 ms step is too small for an H200. Python dispatch, many small kernels, the 2 host
  syncs left in SimOTA, the optimizer, EMA and the gradient all-reduce dominate. Evidence: batch 32/GPU gives 1.28×
  more img/s on the same GPUs, and 8× L40S matches 8× H200 at batch 16.
- **Worker threads.** Each loader worker uses 250–300% CPU despite `OMP_NUM_THREADS=1`, probably OpenCV's own
  thread pool. Next cheap test (tier A): `cv2.setNumThreads(0)` in `worker_init_fn`.
- **Levers, in order:** bigger batch per GPU at the same total batch (fewer GPUs, so the recipe is unchanged);
  CUDA graphs or `torch.compile(mode="reduce-overhead")`, which need fixed shapes per multiscale size; and the
  thread fix above.

## Other data loaders reviewed (2026-11-27, read only, nothing run)

- **ortizeg/object-detection-training** (Apache-2.0; no Ultralytics-derived data code): slower per core.
  - PIL decode and resize; a pandas filter + `iterrows` per sample; deepcopy of cached images; float32 transport.
  - Its mosaic partner cache (40 per worker, random eviction, credited to RTMDet) is the same idea as
    `decode_reuse`. Random eviction instead of FIFO is the one thing to borrow. RTMDet reports it accuracy-neutral;
    no speed gain.
- **Deci-AI super-gradients** (Apache-2.0): simpler and more CPU-hungry.
  - Every partner is decoded fresh; HSV goes BGR→HSV→BGR; float32 is cast in the worker and again in collate.
  - No image cache, no GPU augmentation, no Objects365 recipe.
  - Two ideas: roll the mosaic/mixup probability before loading partners (no gain at prob 1.0); resize on load,
    so the reuse pool holds 640-scale images (a few % of CPU, estimate).
- **Not adopted:** reduced-size JPEG decode (no gain at Objects365 sizes, see the pass-2 cache table) and
  re-encoded JPEG caches (lossy; little size saving for Objects365).

## Considered: super-gradients `YoloXFastDetectionLoss` (2026-11-27) — not adopted, revisit only on evidence

Source: `src/super_gradients/training/losses/yolox_loss.py` (Apache-2.0).
- **What it is:** SimOTA vectorised over sparse (gt, anchor) candidate pairs for the whole batch. Their recipe
  reports YOLOX-S 41.7 h → 18.4 h (2.27×).
- **Why not now:**
  1. `simota_batch` + `cls_cost_algebra` + `simota_nosync` already removed the same bottleneck: SimOTA + loss went
     113 → 19 ms (16% of the step). Even a free loss can't beat that, and on the H200 the step is overhead-bound.
     Their matcher uses `nonzero`, `unique` and sorts, which also synchronise with the host.
  2. It isn't equivalent, by its own docstring. Dynamic k and candidates are taken per GT only, and a
     `dynamic_ks_bias=1.1` compensates. Ours gives identical targets on 16/16 real batches; theirs would be tier C.
  3. Assignment changes haven't helped here: OTA, TAL, QFL and IoU objectness all failed to beat E1 on basketball
     test (foundation-finetune gap study).
- **Worth keeping in mind:**
  - The sparse candidate layout uses less memory than our padded `[B × G_max × A]` tensors at 300 labels per
    mosaic. Write our own sparse version of the exact algorithm if memory limits batch 64.
  - `obj_loss_fix` (objectness normalised by the number of anchors) is a loss change. Under the Objects365 decision above, it
    is screened by a COCO fine-tune from the Objects365 checkpoint.
- **Revisit if:** a profile of the multi-GPU step shows loss/assignment above about 25%. Then run a 30-min A/B of
  their matcher vs ours on real Objects365 batches (~$1 on a 4090) before any accuracy run.

## 8× RTX 4090 on the Objects365 sample (2026-11-27; `results/rtx4090_bench/`, `tools/perf/box_4090.sh`)

Same setup as the 8× L40S rows: `exps/perf/yolox_m_objects365.py` (max_labels 50), patch0+1 (69,519 images), batch
16/GPU, fp16, uncached JPEGs, about 260 iterations. Box: 8× RTX 4090 24 GB, EPYC 7663 (224 vCPUs), 504 GB RAM, $4.21/h,
rented 14 min (about $1).

| run | img/s | data_time | peak RAM | GPU memory |
|---|---:|---:|---:|---:|
| gated flags, 8 workers/GPU | 1,174 | 0.001 s | 49 GB | 22.2 / 24.6 GB |
| + decode_reuse, 8 workers | 1,020 | 0.001 s | 52 GB | 22.2 GB |
| + decode_reuse, 12 workers | 1,274 | 0.001 s | 65 GB | 22.2 GB |

- **Noise:** each run has only about 25 logged iterations, and the multiscale size changes every 10. Normalised to 640 px
  the three runs are 1,087 / 1,089 / 1,139 img/s, so `decode_reuse` shows no clear gain here. The L40S log, normalised the
  same way, is 890 img/s, so the 4090 node is about 1.2× faster. CPU was about 96% idle.
- **Cost per Objects365 epoch** (1,737,289 images):

  | node | $/h | img/s | min/epoch | $/epoch | 20 epochs |
  |---|---:|---:|---:|---:|---:|
  | 8× RTX 4090 | 4.21 | 1,174 | 24.7 | 1.73 | ≈ $35 |
  | 8× L40S | 9.60 | 1,143 | 25.3 | 4.05 | ≈ $81 |
  | 8× H200 | 38.12 | 1,530 | 18.9 | 12.02 | ≈ $240 |

- **Caveats before a 4090 pretrain:**
  - GPU memory is tight: 22.2 of 24.6 GB at batch 16 with max_labels 50. Run the real pretrain config (max_labels 300,
    crowded mosaics) for about 50 iterations before committing.
  - Only one 8× 4090 offer met reliability ≥ 0.98, 64+ cores and 500 Mbps down.
  - These are sample numbers, not the full 1.74 M images.

**Decision update:** the preferred Objects365 node is 8× RTX 4090, about $35 for 20 epochs, if the max_labels 300 memory
check passes. 8× L40S (48 GB) is the fallback, at about $81.

## Pass 5: five small host-side fixes (2026-11-27; PRs #12–#16; `results/pass5_gpu/`, `tools/perf/box_pass5.sh`)

Measured on 2× RTX 4090 (torch 2.5.1+cu124, 64 vCPUs). V = `perf all` minus these five flags and `compile`. Step
times are `bench_step.py` medians over 220 iterations after 110 warm-up, E1 basketball recipe at 800 ±5×32, batch 16.

| PR | flag | tier | GPU tests | equivalence vs V (60 steps) | measured |
|---|---|---|---|---|---|
| #13 | `worker_cv2_threads` | A | (needs 2+ GPUs; run below) | n/a | 2-GPU run, 8 workers/GPU: threads per loader worker **63 → 3**, CPU per worker about 135% → 55%, 216 → 222 img/s (+3%; this box is GPU-bound) |
| #12 | `head_nosync` | A | all pass (one test fixed) | losses and weights identical | 112.6 → 111.7 ms (−0.9%) |
| #16 | `fused_optim_amp` | B | all pass, incl. no host sync in `scaler.step` and a 60-step inf-skip regression | AdamW identical (V already fuses AdamW); SGD loss within 0.34% (orig vs V: 0.39%) | AdamW 112.6 → 112.2 ms; SGD 113.5 → 112.2 ms (−1.1%) |
| #14 | `host_label_counts` | A | all pass (one test fixed) | identical | 112.6 → 112.8 ms (noise) |
| #15 | `lean_ckpt` | A | n/a | training unchanged; resume from `latest` works | 20 basketball epochs with `save_interval 5`: **103 → 90 s (−13%)**; run folder 3.2 → 1.3 GB; best/epoch_N 304 → 101 MB |
| all five | | | | AdamW identical | AdamW 110.8 ms (−1.6%); SGD 111.4 ms (−1.9%) |

- **Why the step gains are small here:** the 1× 4090 basketball step is GPU-bound. Also, `bench_step` synchronises at
  every step boundary, so it can't show the host running ahead into the next step, which is the point of removing syncs.
  The target case is the 8-GPU Objects365 step (35–40% GPU utilisation on 8× H200); measure it there.
- **Bug found and fixed in #16:** torch 2.5.1 fused SGD creates momentum buffers with `torch.empty_like` and relies on
  the kernel's first step to fill them. When GradScaler skips that step (inf/nan at the initial 65536 scale), the
  uninitialised buffers are kept and the next step turns the weights NaN (seen within 3 steps). The fix pre-creates zero
  buffers before each step; SGD with dampening ≠ 0 stays unfused.
- **Multi-GPU hang reproduced:** both 2-GPU runs (with V only, none of these five flags) hung after their final
  evaluation. Rank 0 waited on a 1-element all-reduce until the NCCL watchdog killed it 30 min later
  (`WorkNCCL(SeqNum=3010, OpType=ALLREDUCE, NumelIn=1 ...)`). This looks like the H200 epoch-4 hang. It reproduces in about
  2 min on 2 GPUs, so it can be bisected over V's flags cheaply. Until then, keep in-training eval off for multi-GPU runs.

## N6: no host sync in a training step (2026-11-27; 2× RTX 4090; `tools/perf/box_n6.sh`)

Box: 2× RTX 4090, 32 vCPUs, torch 2.5.1+cu124. V = the 8× L40S audit set without the tier-C `decode_reuse`. E1
basketball recipe (800 ±5×32, AdamW), 16 images/GPU. Raw: `foundation-finetune/results/yolox/n6_gpu/`.

Where the three remaining syncs came from (sync debug mode "error" + autograd anomaly mode names the forward op
behind a backward sync):

| sync | cause | flag | tier |
|---|---|---|---|
| 2 in backward | `torch.prod(·, 1)` in `IOUloss` (`area_p`, `area_i`): its backward checks the input for zeros on the host | `iou_nosync`: areas as w·h | B (forward bit-identical, gradients equal up to rounding) |
| 1 in SimOTA | `fg_flat.nonzero()`: the number of positives sets every later shape | `dense_loss`: targets for every anchor + positive mask, masked dense losses, `num_fg` stays on the device | B (summation order) |
| 4–5 per logging step | one `.item()` per loss meter, again in the tensorboard call | `log_onesync`: one stacked copy | A |

| check | without N6 (V) | with N6 |
|---|---:|---:|
| host syncs per step, real 2-GPU run (rank 0, 20 iterations after 30 warm-up) | 3.50 | **0.15** (= the one copy on logging steps) |
| 2-GPU step, 300 iterations, order V→N6 | 262.8 ms | 254.0 ms (−3.3%) |
| same, order N6→V | 257.7 ms | 250.7 ms (−2.7%) |
| profiled step, fixed 800 px | 232.6 ms, 53.7 ms waiting on syncs, GPU busy 89% | 223.8 ms, 0 ms waiting |
| 60-step equivalence (`bench_step --check`, 1 GPU) | — | `log_onesync` bit-identical; all three: max loss diff 6.2% after 60 AdamW steps |
| same, first 5 steps | upstream vs V: 0.44% | `iou_nosync` 0.10%, `dense_loss` 0.13%, all three 0.19% |
| 1-GPU step (`bench_step`) | 139 ms | 139 ms |
| 20-epoch basketball val AP, seeds 0/1/2 (final epoch) | 62.1 / 60.8 / 60.2 (mean 61.0) | 62.0 / 61.1 / 61.4 (mean 61.5) |
| same, best epoch | 62.07 / 61.65 / 61.28 (61.67) | 61.98 / 61.56 / 61.41 (61.65) |

- **Accuracy: a tie** (best-epoch means 61.67 vs 61.65 on 96 val images). Over 60 AdamW steps the losses drift apart
  as rounding differences compound (upstream vs V does the same, more), but the first 5 steps agree to 0.2% and the
  trained models match.
- **Speed: −3% on 2× 4090**, where the step is already 89% GPU-busy. The 8-GPU Objects365 step (24% idle, see the
  step audit above) is where the gain should be larger; measure it at the next Objects365 run, not on a dedicated box.
- **What it unlocks:** with `dense_loss` the head's tensor shapes no longer depend on the data (only `G`, the
  per-batch max gt count, from host counts), and nothing in the step waits on the GPU. That is N7's prerequisite.
- `dense_loss` costs `[B·A, C]` element-wise work for the class loss: negligible at C = 5–80; at C = 365 (Objects365)
  it is ~49 M elements per step at 640, ~0.2 GB fp32 for the targets. Check memory at the first Objects365 use.
- Tooling: `YOLOX_AUDIT_SPAN_EPOCHS=1` lets audit runs cross epoch boundaries (basketball is 15 iterations/epoch, which
  had ended every audit before its warm-up).
- Verdict: adopt all three into the default set. Next: N7.
