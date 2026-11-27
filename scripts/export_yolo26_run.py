#!/usr/bin/env python3
"""Export one YOLO26m taxonomy-ablation run like the published YOLO26m ONNX.

Same call as yolo26-basketball-training/scripts/export_onnx.py
(format=onnx, imgsz=640, simplify=True, dynamic=True; ultralytics 8.4.21), but the
labels_mapping.json is written from the model's own names (the repo script
hardcodes the 10 classes, which would mislabel the merged5 runs). Schema matches
the published file: {"id_to_name": {...}}.

Checks the exported signature against the published YOLO26m ONNX
(images [batch,3,height,width] -> output0 [batch,300,6], end2end).

Usage: python export_yolo26_run.py <run_dir> <published_onnx>
"""

import json
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def signature(path):
    # imported lazily: ultralytics' export step installs its own pinned onnx deps
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    io = [
        (t.name, [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim])
        for t in list(m.graph.input) + list(m.graph.output)
    ]
    meta = {p.key: p.value for p in m.metadata_props}
    return [(o.domain, o.version) for o in m.opset_import], io, meta


def main():
    run_dir = Path(sys.argv[1])
    published = Path(sys.argv[2])
    best = run_dir / "weights" / "best.pt"

    model = YOLO(str(best))
    model.export(format="onnx", imgsz=640, simplify=True, dynamic=True)
    out = run_dir / f"yolo26m_basketball_{run_dir.name}_640.onnx"
    shutil.copy2(best.with_suffix(".onnx"), out)

    names = {str(k): v for k, v in model.names.items()}
    with open(run_dir / "labels_mapping.json", "w") as f:
        json.dump({"id_to_name": names}, f, indent=2)

    opset, io, meta = signature(out)
    p_opset, p_io, p_meta = signature(published)
    if (opset, io) != (p_opset, p_io):
        raise SystemExit(f"SIGNATURE MISMATCH: {opset} {io} vs published {p_opset} {p_io}")
    for k in ("version", "imgsz", "end2end", "task", "stride", "args"):
        if meta.get(k) != p_meta.get(k):
            raise SystemExit(f"METADATA MISMATCH {k}: {meta.get(k)} vs published {p_meta.get(k)}")
    print(f"{run_dir.name}: ok opset={opset} io={io} names={list(names.values())}")


if __name__ == "__main__":
    main()
