"""Per-clip merged5 test mAP for the taxonomy-ablation manifest.

Reuses run_benchmark's own end2end scorer (same preprocessing, remap, dedupe,
compute_metrics); only adds a split of the 94 test frames by source clip.
Run from the object-detection-eval repo root.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
import run_benchmark as rb  # noqa: E402

from object_detection_eval.data.clips import clip_key  # noqa: E402
from object_detection_eval.data.coco_gt import load_coco_gt  # noqa: E402
from object_detection_eval.data.taxonomy import resolve_taxonomy  # noqa: E402
from object_detection_eval.metrics.detection_map import compute_metrics  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
args = argparse.Namespace(
    data_root=rb._DEFAULT_DATA_ROOT,
    source_repo=rb._DEFAULT_SOURCE_REPO,
    yolox_root=rb._DEFAULT_YOLOX_ROOT,
    providers=rb._DEFAULT_PROVIDERS,
)
manifest = rb.load_manifest(a.manifest)
name_to_id, id_to_name = resolve_taxonomy("merged5")
gt_map = load_coco_gt(args.data_root / "test" / "_annotations.coco.json", name_to_id)
clips = defaultdict(list)
for f in gt_map:
    clips[clip_key(f)].append(f)

out = {"clips": {k: len(v) for k, v in clips.items()}, "models": {}}
for entry in manifest.models:
    pred = rb._score_end2end(entry, args, gt_map, name_to_id, True)
    row = {"all": compute_metrics(gt_map, pred, id_to_name)["mAP_50_95"]}
    for k, files in clips.items():
        row[k] = compute_metrics(
            {f: gt_map[f] for f in files}, {f: pred[f] for f in files}, id_to_name
        )["mAP_50_95"]
    out["models"][entry.name] = {k: float(v) for k, v in row.items()}
    print(entry.name, json.dumps(out["models"][entry.name]), flush=True)
a.out.write_text(json.dumps(out, indent=2))
