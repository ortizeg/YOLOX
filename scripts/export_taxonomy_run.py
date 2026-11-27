#!/usr/bin/env python3
"""Export one taxonomy-ablation run to ONNX exactly like the published YOLOX-M.

Export settings were established by reproducing the published
yolox_m_basketball_640.onnx byte-for-byte from its best_ckpt.pth: legacy
TorchScript exporter (scripts/export_onnx_legacy.py), opset 11, decode in graph,
no NMS, no onnxsim, test_size overridden to 640.

Also writes labels_mapping.json in the published schema and checks the ONNX
input/output signature (and, for 10-class runs, equality with the published one).

Usage:
    PYTHONPATH=. python scripts/export_taxonomy_run.py --tag 10c_seed0
"""

import argparse
import json
import os
import subprocess
import sys

import onnx

ROOT = "training_results/taxonomy_ablation"
PUBLISHED = "training_results/basketball_m/yolox_m_basketball_640.onnx"
EXPS = {
    "10c": "exps/default/yolox_m_basketball_adamw.py",
    "5c": "exps/default/yolox_m_basketball_merged5_adamw.py",
}
DATA = {"10c": "datasets/basketball", "5c": "datasets/basketball_merged5"}


def signature(path):
    m = onnx.load(path, load_external_data=False)
    io = [
        (t.name, [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim])
        for t in list(m.graph.input) + list(m.graph.output)
    ]
    return m.ir_version, [(o.domain, o.version) for o in m.opset_import], io


def labels_mapping(train_json):
    with open(train_json) as f:
        cats = sorted(json.load(f)["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    return {
        "num_classes": len(names),
        "class_names": names,
        "id_to_name": {str(i): n for i, n in enumerate(names)},
        "name_to_original_id": {c["name"]: c["id"] for c in cats},
        "original_id_to_contiguous_id": {str(c["id"]): i for i, c in enumerate(cats)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="e.g. 10c_seed0")
    args = p.parse_args()

    taxo = args.tag.split("_")[0]
    run_dir = os.path.join(ROOT, args.tag)
    onnx_path = os.path.join(run_dir, f"yolox_m_basketball_{args.tag}_640.onnx")
    cmd = [
        sys.executable, "scripts/export_onnx_legacy.py",
        "-f", EXPS[taxo],
        "-c", os.path.join(run_dir, "best_ckpt.pth"),
        "--output-name", onnx_path,
        "--decode_in_inference", "--no-onnxsim",
        "test_size", "(640,640)",
    ]
    with open(os.path.join(run_dir, "export.cmd"), "w") as f:
        f.write(" ".join(cmd) + "\n")
    subprocess.run(cmd, check=True, capture_output=True)

    lm = labels_mapping(os.path.join(DATA[taxo], "annotations", "train.json"))
    with open(os.path.join(run_dir, "labels_mapping.json"), "w") as f:
        json.dump(lm, f, indent=2)

    ir, opset, io = signature(onnx_path)
    n = lm["num_classes"]
    expected = [("images", [1, 3, 640, 640]), ("output", [1, 8400, 5 + n])]
    if (ir, opset, io) != (6, [("", 11)], expected):
        raise SystemExit(f"SIGNATURE MISMATCH {args.tag}: ir={ir} opset={opset} io={io}")
    if taxo == "10c" and signature(PUBLISHED) != (ir, opset, io):
        raise SystemExit(f"10c signature differs from published: {signature(PUBLISHED)}")
    print(f"{args.tag}: ok ir={ir} opset={opset} io={io} classes={lm['class_names']}")


if __name__ == "__main__":
    main()
