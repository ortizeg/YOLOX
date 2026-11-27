# Taxonomy ablation — 10-class vs merged 5-class training

**Question:** does training on the 5-class merged taxonomy change test accuracy versus
the published 10-class models, changing nothing except the taxonomy?
**Answer: no.** On the merged5 test set, 5-class training is within seed noise of
10-class training for both architectures: YOLOX-M **+0.008** mAP@50:95 (0.6725 vs 0.6649),
YOLO26m **+0.002** (0.7141 vs 0.7119), 3 seeds per arm. See [Test results](#test-results-object-detection-eval)
and [Conclusions](#conclusions).

Training and export were done without test labels (test annotations and test images
were never copied to the training box); test mAP was computed afterwards by
`object-detection-eval`, outputs in [`eval/`](eval/).

Runs: `{10c,5c}_seed{0,1,2}/` (YOLOX-M), `yolo26m/{10c,5c}_seed{0,1,2}/` (YOLO26m).

## Step 0 — provenance of the published YOLOX-M (`training_results/basketball_m/`)

| Item | Answer | How established |
|---|---|---|
| Exp | `exps/default/yolox_m_basketball_adamw.py` @ `64c55e2` (only commit of the file) | git |
| Train / test size | **800 / 800** (multiscale ±5×32 ⇒ 640–960 in aug epochs). Not 640. | exp file; the 640 ONNX is a re-export of the same weights |
| Epochs | 200 (best ckpt at `start_epoch=184`, val AP 0.6297) | ckpt keys |
| Batch | 16 (inferred) | final lr 5e-6 = 0.05·(0.0004/64·16); AdamW step 5331 ≈ 184·29 |
| fp16 | yes (inferred) | 5 skipped optimizer steps (GradScaler); S runs logged `fp16=True` |
| Pretrained | **not recorded** — assumed official `yolox_m.pth` (0.1.1rc0) | sibling S runs used `/workspace/yolox_s.pth` |
| Seed | **not recorded** — most likely `None` | sibling S logs show `seed None` |
| GPU | **not recorded** | no M train log survives |
| 640 ONNX export | legacy TorchScript exporter, opset 11, IR 6, `--decode_in_inference` (decode in graph), **no NMS**, `--no-onnxsim`, `test_size "(640,640)"`, torch 2.10.0 (`.venv`) | **reproduced byte-for-byte** (both the 640 and 800 published ONNX) with `scripts/export_onnx_legacy.py` |

Notes: `torch>=2.9` defaults `torch.onnx.export` to the dynamo exporter (opset 18, external
`.data`), so the legacy path must be forced. The eval registry's "fine-tuned at 640×640"
and the blog's "every model trained at 640" are wrong for YOLOX-M.

## Step 1 — datasets (`scripts/build_basketball_datasets.py`)

The original 10-class YOLOX jsons were never copied off the old training box, so both
taxonomies were rebuilt from the Roboflow export (`data/basketball-player-detection-3`,
v1): 10-class = drop zero-annotation `basketball` (id 0), shift ids −1 (matches the
published `labels_mapping.json` exactly); merged5 = `player ← player + 4 player-*`,
`ball ← ball + ball-in-basket`, `referee`, `rim`, `number`. Only `category_id` differs
between the two; images are symlinked. Layout is `annotations/{train,valid,test}.json`
(what YOLOX's `COCODataset` reads).

Full output: [`step1_assertions.txt`](step1_assertions.txt). All checks passed:

| | train | valid | test |
|---|---|---|---|
| identical image ids/order 10c vs 5c | 464 | 96 | 94 |
| annotations conserved | 9,552 | 1,953 | 1,980 |
| player / ball / referee / rim / number | 4556/465/1221/455/2855 | 951/97/287/96/522 | 944/96/280/93/567 |
| same-class IoU>0.9 pairs, 10c → 5c | 0 → 1 | 0 → 1 | 0 → 1 |

The three post-merge pairs are one player double-labelled (`player`+`player-shot-block`
train IoU 0.983; `player`+`player-jump-shot` valid 0.987, test 0.990). Kept, not removed;
under merged5 the test pair is a duplicate GT box.
Real split sizes are 464/96/94 images (the "465/98/95" in eval `training-runs.md` is wrong;
465 is the train `ball` count).

## Step 2 — exp diff

`exps/default/yolox_m_basketball_merged5_adamw.py` vs the published exp — only these lines:

```diff
-        self.num_classes = 10
-        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
+        self.num_classes = 5
+        self.exp_name = "yolox_m_basketball_merged5_adamw"
-        self.data_dir = "datasets/basketball"
+        self.data_dir = "datasets/basketball_merged5"
```

## Step 3 — YOLOX-M training

- **Code:** YOLOX `64c55e2` (branch `dev/phase4-distill`) + the merged5 exp and
  `scripts/{build_basketball_datasets.py,run_taxonomy_ablation.sh,pull_taxonomy_run.sh,export_onnx_legacy.py,export_taxonomy_run.py}`
  (uncommitted at training time, since committed as `84a84f1` on `feat/taxonomy-ablation`);
  `tools/export_onnx.py` carries a pre-existing uncommitted `_export → export` edit
  (training code is unmodified).
- **Env:** pixi (`pixi.toml`), torch 2.5.1+cu124, cuDNN 9.1.
- **GPU:** vast.ai instance 58021864, 1× RTX 4090 (48 GB variant), driver 570.153.02, 16 vCPU.
- **Pretrained:** official `yolox_m.pth` (0.1.1rc0), sha256 `60076992b32da82951c90cfa7bd6ab70eba9eda243e08b940a396f60ac2d19b6`.
- **Command** (identical to the published run except `-f`, `seed N`, and `-expn`, which only names the output dir):

```
pixi run python -m yolox.tools.train -f exps/default/<exp>.py -expn <tag> -d 1 -b 16 --fp16 \
  -c /workspace/yolox_m.pth --cache ram -o --logger tensorboard seed <N>
```

Checkpoint selection: unchanged (best val AP, YOLOX default). Per-run `<tag>.cmd`,
`<tag>.stdout`, `train_log.txt` are in each run dir.

| Run | best val AP@50:95 | best epoch | wall-clock | best_ckpt sha256 |
|---|---|---|---|---|
| 10c_seed0 | 0.6173 | 175 | 61.5 min* | `980f517eca017151…` |
| 10c_seed1 | 0.6425 | 115 | 32.2 min | `d9ae26087cbad35a…` |
| 10c_seed2 | 0.6285 | 199 | 29.8 min | `0064f1e91effc99e…` |
| 5c_seed0 | 0.6995 | 174 | 33.1 min | `606fcbe17ccc411d…` |
| 5c_seed1 | 0.6987 | 187 | 31.4 min | `45e08cee42ef7064…` |
| 5c_seed2 | 0.7002 | 180 | 34.0 min | `cbd494e69007f397…` |
| *published* | *0.6297* | *184* | — | — |

10c mean val AP 0.6294 (published 0.6297) — the rerun reproduces the published model.
**5c and 10c val APs are not comparable** (each averages over its own class set); the
comparison is the merged5 test mAP from `object-detection-eval`.

\* 10c_seed0 is the only run that also wrote YOLOX's per-eval history checkpoints
(`save_history_ckpt`, ~20 GB); later runs delete them in the background (write-only
files, training unaffected), so its wall-clock is not comparable.

### Caveats

- **Seeding is partial.** `seed N` (passed as an exp opt; YOLOX's `tools/train.py` has no
  `--seed` flag, and the value arrives as the string `'N'`) seeds model init, sampler order
  and multiscale size choice and turns on cuDNN determinism, but YOLOX reseeds every
  dataloader worker from `uuid4()` (`yolox/data/dataloading.py:110`), so mosaic/mixup/HSV
  augmentation is random. Runs are not bit-reproducible; the 3-seed spread is the noise.
- **Disk-full incident.** The first attempt at `5c_seed0`, `10c_seed1`, `5c_seed1` crashed
  when history checkpoints filled the 60 GB disk; YOLOX's `@logger.catch` exited 0 so the
  runner initially recorded them as done. Those runs were discarded and rerun from scratch;
  their logs are in [`failed_diskfull/`](failed_diskfull/). The runner now fails loudly on
  any traceback.

## Step 4 — YOLOX-M export

Every best checkpoint exported with the exact Step 0 settings via
`scripts/export_taxonomy_run.py` (command in each run's `export.cmd`), which also writes
`labels_mapping.json` (published schema) and asserts the signature:

- all: IR 6, opset 11, input `images [1,3,640,640]`, output `output [1,8400,5+nc]`,
  decode in graph, no NMS
- 10c: output `[1,8400,15]`, signature identical to the published 640 ONNX; 10c
  `labels_mapping.json` content identical to the published one
- 5c: output `[1,8400,10]`, classes `player, ball, referee, rim, number`

| Run | ONNX | sha256 |
|---|---|---|
| 10c_seed0 | `yolox_m_basketball_10c_seed0_640.onnx` | `e626b52b60607979…` |
| 10c_seed1 | `yolox_m_basketball_10c_seed1_640.onnx` | `583649decffe49b6…` |
| 10c_seed2 | `yolox_m_basketball_10c_seed2_640.onnx` | `961f58f043a14630…` |
| 5c_seed0 | `yolox_m_basketball_5c_seed0_640.onnx` | `79324a3b683739bc…` |
| 5c_seed1 | `yolox_m_basketball_5c_seed1_640.onnx` | `08597b7861ddd01b…` |
| 5c_seed2 | `yolox_m_basketball_5c_seed2_640.onnx` | `affa4c5cff6ee633…` |

Eval note: these are the same family as `yolox_m_basketball_640.onnx` — score with the
existing YOLOX inferencer (top-left letterbox pad-114, BGR, raw 0–255, external NMS).

## YOLO26m

Added after the YOLOX arm, on the same box, so the taxonomy result can be checked on a
second architecture (YOLO26m is the top 5-class model in the published comparison).

### Provenance of the published YOLO26m

From `train_args` inside the published `best.pt`
(`gs://deep-ego-model-training/ego-training-data/basketball-data/logs/yolo26m-basketball-training/20260305_182204/best.pt`,
read-only):

| Item | Value |
|---|---|
| Code | `yolo26-basketball-training` @ `80e20eb` (`scripts/train.py`), image `ultralytics/ultralytics:latest` |
| Ultralytics | 8.4.21 (commit `83471a5`) |
| Init | `yolo26m.pt` (COCO) |
| epochs / batch / imgsz | 300 / 8 / 640 |
| seed / deterministic / amp | 0 / True / True |
| optimizer | `auto` → resolved to **AdamW, lr0 0.000714, momentum 0.9, warmup_bias_lr 0** |
| patience | 100 (never triggered; best epoch 279) |
| best val mAP50-95 | 0.6848 |
| GPU, torch | not recorded (Vertex AI) |

**LR confound and how it was removed.** Under `optimizer=auto` with fewer than 10,000
nominal iterations (here ⌈464/64⌉·300 = 2,400), Ultralytics 8.4.21 picks AdamW with
`lr0 = round(0.002·5/(4+nc), 6)`: 0.000714 at nc=10 (matches the published `results.csv`
peak 0.000707) but **0.001111 at nc=5**. Rerunning the identical command would change
taxonomy and LR together. With the user's sign-off, both arms pin the optimizer `auto`
resolved to for the published run: `optimizer=AdamW, lr0=0.000714, momentum=0.9,
warmup_bias_lr=0.0`. This follows the exact same code path as `auto` (AdamW,
betas (0.9, 0.999), same param groups / weight decay); only the nc-dependence is removed.
A full diff of the 10c_seed0 `args.yaml` against the published `train_args` differs only
in `optimizer/lr0/momentum` (the pinned values; the published 0.01/0.937 were ignored by
`auto`) and in paths/run name.

### Data (`scripts/build_yolo26_datasets.py`)

Converted with the YOLO26 repo's own `convert_coco_to_yolo.convert_split`, unchanged: the
10-class arm uses its name-based `build_category_remap` (exactly as published); the 5-class
arm the same converter with a name → merged5 remap. Train/valid only. Output:
[`yolo26m/yolo26_data_build.txt`](yolo26m/yolo26_data_build.txt) — all checks passed
(464/96 label files per arm, 9,552/1,953 annotations conserved, merged5 counts as in
Step 1, and every 5-class label line equals the 10-class line with only the class id
remapped).

### Training

- **Env:** venv, ultralytics 8.4.21, torch 2.10.0+cu128 (the cu130 wheel pip picked by
  default cannot run on driver 570; the published run's torch version is unknown), onnx
  1.23.2, onnxslim 0.1.98. Full `pip freeze` in [`yolo26m/environment.txt`](yolo26m/environment.txt).
- **GPU:** same vast.ai box (RTX 4090 48 GB, driver 570.153.02).
- **Pretrained:** `yolo26m.pt` auto-downloaded by 8.4.21, sha256 `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`.
- **Runner:** `scripts/run_yolo26_taxonomy_ablation.sh`; exact per-run call in
  `yolo26m/<tag>/<tag>.py`:

```python
YOLO("yolo26m.pt").train(
    data="<10c|5c>/data.yaml", epochs=300, imgsz=640, batch=8,
    project=..., name="<tag>", exist_ok=True, verbose=True,
    seed=<N>, deterministic=True,
    optimizer="AdamW", lr0=0.000714, momentum=0.9, warmup_bias_lr=0.0,
)
```

Checkpoint selection unchanged (Ultralytics `best.pt` = max fitness; in 8.4.21 fitness =
val mAP50-95).

| Run | best val mAP50-95 | best epoch | epochs run | wall-clock | best.pt sha256 |
|---|---|---|---|---|---|
| 10c_seed0 | 0.6679 | 170 | 270 † | 42.5 min | `ca0c11b3684057b1…` |
| 10c_seed1 | 0.6614 | 202 | 300 | 44.0 min | `a24e2651c9387d27…` |
| 10c_seed2 | 0.6675 | 266 | 300 | 43.7 min | `25903e2c90234cda…` |
| 5c_seed0 | 0.7356 | 222 | 300 | 43.8 min | `c16ca0cb153688c3…` |
| 5c_seed1 | 0.7318 | 225 | 300 | 44.5 min | `93a978adff6ffb8e…` |
| 5c_seed2 | 0.7361 | 235 | 300 | 44.5 min | `9345631a53ad9869…` |
| *published 10c* | *0.6848* | *279* | *300* | — | — |

(Exact hashes in each run's `sha256.txt`.)

† `patience=100` (the published setting) early-stopped 10c_seed0; the published run never
hit it.

**The 10c reruns sit ~0.019 below the published model** (mean 0.6656, tight spread
0.6614–0.6679, vs 0.6848). The full arg set is identical apart from the pinned optimizer,
which equals what `auto` chose, so the likely cause is the environment (torch build, GPU,
cuDNN — `deterministic=True` is only deterministic within one environment). Both arms
share that environment, so the 10c-vs-5c comparison is unaffected, but the reruns, not
the published model, are the right 10c baseline. Val mAP is again **not comparable
across taxonomies**.

### Export

`scripts/export_yolo26_run.py`, run on the box with the training env: the same call as the
repo's `scripts/export_onnx.py` (`format=onnx, imgsz=640, simplify=True, dynamic=True`),
except `labels_mapping.json` is written from the model's own names (the repo script
hardcodes the 10 classes). Schema matches the published file (`{"id_to_name": …}`).
Asserted for every run against the published `training_results/yolo26m/model.onnx`:
opset 22, `images [batch,3,height,width]` → `output0 [batch,300,6]` (end-to-end,
NMS-free), and identical metadata `version / imgsz / end2end / task / stride / args`.
ONNX: `yolo26m/<tag>/yolo26m_basketball_<tag>_640.onnx` (hashes in `sha256.txt`).

Eval note: score with the existing YOLO26 inferencer (centered letterbox, RGB, /255).

## Test results (object-detection-eval)

**Protocol.** `object-detection-eval` @ `8751c1d`, unmodified, via its own
`scripts/run_benchmark.py --mode end2end --taxonomy merged5` with a 14-model manifest
([`eval/taxonomy_ablation_manifest.yaml`](eval/taxonomy_ablation_manifest.yaml); kept
outside the eval repo). Same path as the published @640 table: each model's own
train-matched letterbox, conf 0.01, CPU onnxruntime, name-based remap to merged5, the
post-remap merged-class dedupe (applied to every model alike), 94 test images. The
"reproduction gate FAILED" line in the log is expected — the gate checks published rank
order, which this manifest does not encode. Outputs: [`eval/merged5.json`](eval/merged5.json)
(mAP@50:95/50/75 + per-class AP50), [`eval/merged5.log.gz`](eval/merged5.log.gz).

The same scoring is committed upstream as a scorer, manifest, results file and generated report
(`benchmarks/basketball/reports/TAXONOMY_ABLATION.md`, ortizeg/object-detection-eval PR #40); its
numbers are identical to the ones below.

**Harness check.** The published models, scored in the same run, reproduce the published
table: YOLO26m 0.7173 (published 0.716), YOLOX-M @640 0.6681 (published 0.672, within the
gate's 0.01 tolerance).

| Model | 10c seed 0 / 1 / 2 | 5c seed 0 / 1 / 2 | 10c mean ± sd | 5c mean ± sd | Δ (5c − 10c) |
|---|---|---|---|---|---|
| YOLOX-M mAP@50:95 | 0.6768 / 0.6463 / 0.6714 | 0.6744 / 0.6627 / 0.6804 | 0.6649 ± 0.0163 | 0.6725 ± 0.0090 | **+0.0076** (Welch p 0.53) |
| YOLOX-M mAP@50 | | | 0.9269 ± 0.0092 | 0.9330 ± 0.0070 | +0.0060 (p 0.42) |
| YOLOX-M mAP@75 | | | 0.7725 ± 0.0158 | 0.7737 ± 0.0170 | +0.0012 (p 0.93) |
| YOLO26m mAP@50:95 | 0.7118 / 0.7025 / 0.7215 | 0.7118 / 0.7179 / 0.7125 | 0.7119 ± 0.0095 | 0.7141 ± 0.0033 | **+0.0021** (p 0.74) |
| YOLO26m mAP@50 | | | 0.9503 ± 0.0096 | 0.9439 ± 0.0045 | −0.0064 (p 0.38) |
| YOLO26m mAP@75 | | | 0.8318 ± 0.0162 | 0.8349 ± 0.0037 | +0.0031 (p 0.78) |

Per-class AP50, mean of 3 seeds (10c → 5c):

| | player | ball | referee | rim | number |
|---|---|---|---|---|---|
| YOLOX-M | 0.952 → 0.940 | 0.799 → 0.833 | 0.990 → 0.988 | 1.000 → 1.000 | 0.894 → 0.903 |
| YOLO26m | 0.979 → 0.976 | 0.869 → 0.846 | 0.968 → 0.969 | 1.000 → 1.000 | 0.935 → 0.928 |

Per-clip mAP@50:95, mean of 3 seeds (Δ 5c − 10c; [`eval/per_clip.py`](eval/per_clip.py)
reuses `run_benchmark`'s scorer and only splits frames by clip, output
[`eval/per_clip.json`](eval/per_clip.json)):

| Clip (frames) | YOLOX-M 10c → 5c | YOLO26m 10c → 5c |
|---|---|---|
| knicks game 4 q1 (33) | 0.6665 → 0.6694 (+0.003) | 0.7267 → 0.7359 (+0.009) |
| magic game 4 q1 (31) | 0.6309 → 0.6446 (+0.014) | 0.6992 → 0.6851 (−0.014) |
| knicks game 1 q1 (30) | 0.7101 → 0.7138 (+0.004) | 0.7162 → 0.7315 (+0.015) |

## Conclusions

1. **Training on the merged 5-class taxonomy does not measurably change test accuracy.**
   The 5c − 10c difference is +0.008 (YOLOX-M) and +0.002 (YOLO26m) mAP@50:95, smaller
   than the seed-to-seed spread of a single arm (YOLOX-M 10c ranges 0.646–0.677) and far
   from significant (p 0.53 / 0.74 across seeds; that test ignores test-set sampling noise,
   which with 3 clips is larger still). If there is an effect, it is below ~0.01–0.02 and
   this setup cannot resolve it.
2. **No consistent direction.** YOLOX-M 5c is nominally ahead overall and on all three
   clips, but YOLO26m's per-clip deltas flip sign (+0.009 / −0.014 / +0.015), mAP@50 goes
   the other way for YOLO26m, and the one class with a visible per-class shift — `ball`
   AP50 — moves +0.034 for YOLOX-M and −0.023 for YOLO26m. These are noise, not a
   taxonomy effect.
3. **The large validation gap was a metric artifact.** 5c val AP is ~0.07 above 10c for
   both architectures (YOLOX-M 0.700 vs 0.629; YOLO26m 0.734 vs 0.666) only because each
   averages over its own class set (the 10-class average also covers the
   `player-*`/`ball-in-basket` action classes, presumably the harder ones; per-class val AP
   was not broken out). Scored on the same 5 classes, the gap is gone.
4. **Taxonomy choice is free on accuracy grounds.** The 10-class model loses nothing at
   5-class evaluation time and keeps the action labels; the 5-class model is equivalent if
   only the 5 classes are needed. The published comparison, which trains on 10 classes
   and scores merged5, is not handicapped by that choice.
5. **Seed variance matters more than taxonomy.** Single-run test mAP of the same 10c
   YOLOX-M config spans 0.031 (0.646–0.677), and the published YOLOX-M (0.668) sits inside
   that range. Gaps of this size between single runs in the published leaderboard should
   be read with that in mind.
6. **YOLO26m reruns match the published model on test.** Their val mAP was ~0.019 below
   published, but on test the 10c reruns average 0.712 vs 0.717 published, within seed
   spread (0.703–0.722). The environment difference flagged under Training mostly
   disappears on test.

## Cost

One vast.ai instance (58021864, RTX 4090, $0.468/h), 2026-10-27 22:53 → 2026-10-28 08:57 UTC
= 10.06 h = **$4.71** (cap $25). Covers env setup, the discarded disk-full attempts,
6 YOLOX-M runs and 6 YOLO26m runs. Instance destroyed after all 12 checkpoints were
pulled and sha256-verified locally.

## Reproduction

```bash
# datasets (YOLOX, laptop)
python scripts/build_basketball_datasets.py --src ../../data/basketball-player-detection-3 --out-root datasets
# YOLOX training (box, repo root)
bash scripts/run_taxonomy_ablation.sh
# YOLOX export (laptop, YOLOX/.venv torch 2.10.0)
PYTHONPATH=. .venv/bin/python scripts/export_taxonomy_run.py --tag <tag>
# YOLO26m datasets + training + export (box)
python build_yolo26_datasets.py --src /workspace/data --converter-dir <yolo26 repo>/scripts --out-root /workspace/yolo26_data
bash run_yolo26_taxonomy_ablation.sh
python export_yolo26_run.py /workspace/yolo26_runs/<tag> <published model.onnx>
```
