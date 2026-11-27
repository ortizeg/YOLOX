# Pass 3: further training speed-ups (2026-11-26)

Independent review of the training path, done after passes 1–2. Seven new opt-in flags; `perf ""` is still the
original code everywhere. CPU suite: 284 passed (29 new tests in `tests/perf/test_perf_pass3.py`). CPU numbers are
from Apple silicon; GPU effects are estimates until benchmarked on the box.

## Implemented

| flag | tier | what | measured (CPU) |
|---|---|---|---|
| `cv_flip` | A | `cv2.flip` instead of copying an `img[:, ::-1]` view (25–40× slower copy) in `_mirror` and mixup | +7.5% samples/s/core at 640 / +19.6% at 800 |
| `preproc_nopad` | A | skip the 114 canvas + copy when the image already fills the input (every mosaic sample) | +5.3% / +16.3% alone |
| `hwc_transport` | A | workers send HWC uint8; permute + cast is one GPU copy (`trainer.to_nchw`); needs `uint8_transport` | +2.5% / +10.1% alone |
| the three together | A | | **+11.3% (640, uncached, Objects365-like) / +29.7% (800, RAM cache, basketball)** |
| `fast_anno` | A | annotation parsing with Python min/max (non-finite values fall back to the numpy path) | dataset start-up 16.9 → 5.1 s per 100 k images (≈ 4.8 → 1.4 min per rank at 1.7 M) |
| `int_indices` | A | sampler yields ints, not 0-d tensors | 0.14 → 0.004 ms per batch |
| `simota_nosync` | B | kmax = 10 (dynamic k ≤ 10 always), one `nonzero`, index gathers; bit-identical on CPU, tier B for CUDA topk ties like `simota_batch` | host syncs 8–10 → 2 per step (GPU est. ≤ 1–2%) |
| `resize_nosync` | A | multiscale size drawn on the host; DDP broadcast over a cached gloo group | est. < 0.5% (1 GPU), 0.5–1% (8 GPUs) |

## Not implemented, ranked

1. **Fewer JPEG decodes per sample** (C): about 60% of per-sample CPU at Objects365 scale is decoding 5 images per
   mosaic+mixup sample. Reusing decoded partners within a worker, GPU decode or reduced decode would give up to
   2–3× data throughput but change results, so each needs an accuracy run.
2. **Objects365 memory** (A, do before the run): every rank keeps the training COCO object (about 25 M annotation
   dicts) and the per-image lists get copied into forked workers as refcounts are touched. Drop `self.coco` after
   parsing and pack annotations into one array.
3. DDP `static_graph`, larger `bucket_cap_mb` (A) and fp16 gradient compression (C): a few % on PCIe-only nodes;
   needs multi-GPU to measure.
4. Workers per GPU: about 78 samples/s/core after this pass vs about 135 img/s per GPU at 118 ms/step, so at least
   2 augmentation cores per GPU plus decode headroom on Objects365.
5. Small: `load_resized_img` no-op resize at the target size (~1%), mosaic canvas fill (2–7% of a sample), EMA
   temporary (< 0.5 ms/step), head grid rebuild (< 1 ms at 365 classes).
6. Flagged for accuracy: `max_labels=120` truncates crowded Objects365 mosaics.

Dropped after measuring: 3-channel `cv2.LUT` (no gain), RAM-cache `deepcopy` (0.01–0.02 ms), separate
channels_last copy (~0.1 ms).

## GPU check (next box)

```bash
V=<validated set from LEDGER.md>; P3D=cv_flip,preproc_nopad,hwc_transport,int_indices
python tools/perf/bench_step.py -f $EXP -c yolox_m.pth --out p3_step.json --sets "$V" "$V,simota_nosync" -- <E1 opts>
python tools/perf/bench_step.py -f $EXP -c yolox_m.pth --check --iters 60 --warmup 0 --ref "$V" \
    --out p3_check.json --sets "$V,simota_nosync" -- <E1 opts>
python tools/perf/bench_data.py -f $EXP --workers 4 8 --batches 60 --out p3_data.json --sets "" data "data,$P3D" -- <E1 opts>
```
