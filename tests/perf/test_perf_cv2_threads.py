"""Pass-5 N1: worker_cv2_threads (tier A).

configure_module() (cv2.setNumThreads(0), no OpenCL) only runs under tools/train.py's __main__. Spawned DDP ranks
and their dataloader workers keep OpenCV's default thread pool (one thread per core). With the flag on, the worker
init fn and the spawned-rank entry apply the same cv2 settings. cv2.setNumThreads(0) means "run cv2 code
sequentially in the calling thread"; cv2.getNumThreads() then reports 1.

cv2 thread state is process-wide, so every check of it runs in a fresh subprocess.
"""

import json
import os
import pickle
import subprocess
import sys

import numpy as np

from yolox.utils import perf

from .conftest import flags
from .test_perf_data import DATA_FLAGS, build, samples

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(code, spec):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("YOLOX_PERF", None)
    if spec:
        env["YOLOX_PERF"] = spec
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


INIT_FN = """
import json, cv2
before = (cv2.getNumThreads(), bool(cv2.ocl.useOpenCL()))
from yolox.data.dataloading import worker_init_reset_seed
worker_init_reset_seed(0)
print(json.dumps({"before": before, "after": (cv2.getNumThreads(), bool(cv2.ocl.useOpenCL()))}))
"""


def test_flag_registered():
    assert perf.TIER["worker_cv2_threads"] == "A"
    assert "worker_cv2_threads" in perf.parse("all") and "worker_cv2_threads" in perf.parse("tierA")


def test_init_fn_flag_on_limits_cv2():
    r = run(INIT_FN, "worker_cv2_threads")
    assert r["after"] == [1, False]


def test_init_fn_flag_off_leaves_cv2_unchanged():
    r = run(INIT_FN, "")
    assert r["after"] == r["before"]


LOADER = """
import json, sys, cv2, torch
from torch.utils.data import Dataset
from yolox.data.dataloading import worker_init_reset_seed

class Probe(Dataset):
    def __len__(self):
        return 4
    def __getitem__(self, i):
        return torch.tensor([cv2.getNumThreads(), int(cv2.ocl.useOpenCL())])

if __name__ == "__main__":
    ctx = sys.argv[1]
    dl = torch.utils.data.DataLoader(Probe(), batch_size=1, num_workers=2, worker_init_fn=worker_init_reset_seed,
                                     multiprocessing_context=ctx)
    seen = sorted({tuple(b[0].tolist()) for b in dl})
    print(json.dumps({"default": cv2.getNumThreads(), "seen": seen}))
"""


def run_loader(ctx, spec, tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(LOADER)
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("YOLOX_PERF", None)
    if spec:
        env["YOLOX_PERF"] = spec
    out = subprocess.run([sys.executable, str(script), ctx], env=env, cwd=ROOT, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_real_workers_see_flag(tmp_path):
    """The flag reaches real dataloader workers (fork inherits the module state, spawn re-reads YOLOX_PERF)."""
    # fork only on Linux (the training boxes): on macOS a forked child calling into cv2's GCD pool aborts
    for ctx in ("fork", "spawn") if sys.platform.startswith("linux") else ("spawn",):
        on = run_loader(ctx, "worker_cv2_threads", tmp_path)
        assert on["seen"] == [[1, 0]], (ctx, on)
        off = run_loader(ctx, "", tmp_path)
        if off["default"] > 1:  # original path: workers keep the full cv2 pool
            assert all(n > 1 for n, _ in off["seen"]), (ctx, off)


def test_pipeline_identical_with_cv2_single_thread(coco_dir):
    """Tier A: cv2 ops give the same pixels sequential vs threaded. Original pipeline in this process (default cv2
    threads) vs the same pipeline in a subprocess after the flag-on worker init (cv2 single-threaded)."""
    with flags(""):
        ref = samples(build(coco_dir), 16, 3000)
    code = f"""
import json, pickle, sys
from yolox.utils import perf
from yolox.data.dataloading import worker_init_reset_seed
import cv2
worker_init_reset_seed(0)
assert cv2.getNumThreads() == 1
perf.set_flags("")  # same data pipeline as the reference; only cv2 threading differs
from tests.perf.test_perf_data import build, samples
out = samples(build({str(coco_dir)!r}), 16, 3000)
sys.stdout.buffer.write(pickle.dumps(out))
"""
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
               YOLOX_PERF="worker_cv2_threads")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, timeout=300)
    assert out.returncode == 0, out.stderr.decode()
    new = pickle.loads(out.stdout)
    for (ri, rl, rinfo), (ni, nl, ninfo) in zip(ref, new):
        assert np.array_equal(ri, ni)
        assert np.array_equal(rl, nl)
        assert rinfo == ninfo


def test_pipeline_identical_flag_on_off(coco_dir):
    """The flag itself changes nothing in the data path (it only acts in worker init / rank entry)."""
    with flags(DATA_FLAGS):
        ref = samples(build(coco_dir), 16, 4000)
    with flags(DATA_FLAGS + ",worker_cv2_threads"):
        new = samples(build(coco_dir), 16, 4000)
    for (ri, rl, _), (ni, nl, _) in zip(ref, new):
        assert np.array_equal(ri, ni) and np.array_equal(rl, nl)


def test_distributed_worker_limits_cv2_when_on():
    """The spawned-rank entry applies the cv2 settings first; CUDA is faked absent so the rank stops at its CUDA
    assert right after (the rank body never runs here). Flag off: cv2 untouched."""
    code = """
import json, cv2, torch
torch.cuda.is_available = lambda: False
import importlib
launch = importlib.import_module("yolox.core.launch")
try:
    launch._distributed_worker(0, None, 1, 1, 0, "gloo", "tcp://127.0.0.1:1", ())
except AssertionError:
    pass
print(json.dumps((cv2.getNumThreads(), bool(cv2.ocl.useOpenCL()))))
"""
    assert run(code, "worker_cv2_threads") == [1, False]
    assert run(code, "") == run(INIT_DEFAULT, "")


INIT_DEFAULT = """
import json, cv2
print(json.dumps((cv2.getNumThreads(), bool(cv2.ocl.useOpenCL()))))
"""
