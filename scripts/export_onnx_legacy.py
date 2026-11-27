#!/usr/bin/env python3
"""Run tools/export_onnx.py with the legacy TorchScript ONNX exporter.

torch>=2.9 defaults torch.onnx.export to the dynamo exporter (opset>=18,
external .data file). The published YOLOX basketball ONNX files were produced by
the TorchScript exporter (opset 11, IR 6, single file), so force dynamo=False.
All arguments are passed through to tools/export_onnx.py unchanged.
"""

import functools
import os
import runpy
import sys

import torch

torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)

if __name__ == "__main__":
    tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
    sys.argv[0] = os.path.join(tools, "export_onnx.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
