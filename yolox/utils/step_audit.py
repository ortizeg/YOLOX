"""Training-step audit hooks: short runs, host-sync audit, stage profile and a GPU memory cap.

Everything here is off unless one of these environment variables is set (``from_env`` then returns None and the
trainer runs its original code path; ``stage`` returns a shared no-op context):

- ``YOLOX_AUDIT_ITERS=N``     stop after N iterations of the first epoch (no checkpoint writes, no evaluation).
- ``YOLOX_SYNC_AUDIT=<path>``  rank 0: after ``YOLOX_SYNC_AUDIT_WARMUP`` (30) iterations, turn on
  ``torch.cuda.set_sync_debug_mode("warn")`` for ``YOLOX_SYNC_AUDIT_STEPS`` (20) iterations and write every host
  sync, deduplicated by call site, with a trimmed stack, to a JSON file.
- ``YOLOX_PROFILE=<dir>``      rank 0: torch.profiler, schedule(wait=30, warmup=5, active=10), CPU+CUDA, with
  ``audit/<stage>`` ranges around the step stages. Writes ``trace_rank0.json`` (chrome trace),
  ``summary.json`` (``summarize_trace``) and ``key_averages.txt``.
- ``YOLOX_MEM_CAP_GB=G``       every rank: ``torch.cuda.set_per_process_memory_fraction(G GiB / total)``.
- ``YOLOX_AUDIT_OUT=<dir>``    where every rank writes ``mem_rank{r}.json`` (peak allocated/reserved, OOM) at exit
  when any of the above is set (default ``<run dir>/audit``).

``python -m yolox.utils.step_audit <trace.json> [out.json]`` re-runs the summary on a saved trace.
"""

from __future__ import annotations

import bisect
import contextlib
import json
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict

from loguru import logger

_NULL = contextlib.nullcontext()
_MODE = 0  # 0: stage() is a no-op; 1: track the current stage (sync audit); 2: also emit profiler ranges
_STACK: list[str] = []
PREFIX = "audit/"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_THIS_FILE = os.path.abspath(__file__)


class _Stage:
    __slots__ = ("name", "rf")

    def __init__(self, name):
        self.name = name
        self.rf = None

    def __enter__(self):
        _STACK.append(self.name)
        if _MODE >= 2:
            import torch

            self.rf = torch.profiler.record_function(PREFIX + self.name)
            self.rf.__enter__()
        return self

    def __exit__(self, *exc):
        if self.rf is not None:
            self.rf.__exit__(*exc)
        if _STACK:
            _STACK.pop()
        return False


def stage(name: str):
    """Context manager around one step stage. A shared ``nullcontext`` unless rank 0 runs a sync audit or a
    profile, so normal training pays one global read per stage."""
    if not _MODE:
        return _NULL
    return _Stage(name)


def current_stage() -> str | None:
    return "/".join(_STACK) if _STACK else None


def _env_int(name, default):
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


def from_env(rank: int, out_dir: str):
    """A ``StepAudit`` when any audit variable is set, else None (the trainer then skips every hook)."""
    keys = ("YOLOX_AUDIT_ITERS", "YOLOX_SYNC_AUDIT", "YOLOX_PROFILE", "YOLOX_MEM_CAP_GB")
    if not any(os.environ.get(k, "").strip() for k in keys):
        return None
    return StepAudit(rank, out_dir)


# --------------------------------------------------------------------------------------------- sync audit
class SyncRecorder:
    """Collects ``set_sync_debug_mode("warn")`` warnings while active, keyed by call site.

    The warnings are raised by torch's C++ handler through Python's ``warnings`` module at the Python line that
    made the call (for backward, at the ``backward()`` call once the engine replays its warnings). ``always``
    is pushed in front of every existing filter (``-W ignore``, the per-location ``default`` registry) and
    ``torch.set_warn_always(True)`` is set, so no repeat is dropped."""

    def __init__(self):
        self.sites: dict[str, dict] = {}
        self.other: dict[str, int] = defaultdict(int)
        self.iter = -1
        self.iters = 0
        self._cm = None
        self._warn_always = None

    def start(self):
        import torch

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("always")
        warnings.showwarning = self._show
        self._warn_always = torch.is_warn_always_enabled()
        torch.set_warn_always(True)
        torch.cuda.set_sync_debug_mode("warn")

    def stop(self):
        import torch

        torch.cuda.set_sync_debug_mode("default")
        if self._warn_always is not None:
            torch.set_warn_always(self._warn_always)
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None

    def next_iter(self):
        self.iter += 1
        self.iters += 1

    def _show(self, message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if "synchroniz" not in text:
            self.other[f"{category.__name__}: {text[:200]}"] += 1
            return
        self.record(text, traceback.extract_stack()[:-1])

    def record(self, text, frames):
        """``frames``: ``traceback.extract_stack()`` order (outermost first)."""
        frames = [f for f in frames if os.path.abspath(f.filename) != _THIS_FILE
                  and not f.filename.endswith(os.sep + "warnings.py")]
        leaf = frames[-1] if frames else None

        def is_own(f):
            return os.path.abspath(f.filename).startswith(_REPO_ROOT + os.sep)

        own = [f for f in reversed(frames) if is_own(f)]

        def fmt(f):
            path = os.path.relpath(os.path.abspath(f.filename), _REPO_ROOT) if is_own(f) else f.filename
            return f"{path}:{f.lineno} {f.name}: {(f.line or '').strip()}"

        site = fmt(own[0]).split(": ", 1)[0] if own else (fmt(leaf).split(": ", 1)[0] if leaf else "?")
        if leaf is not None and own and leaf is not own[0]:
            site += f" -> {leaf.name}"
        s = self.sites.get(site)
        if s is None:
            s = self.sites[site] = {"count": 0, "iters_seen": [], "stage": current_stage(), "message": text,
                                    "leaf": fmt(leaf) if leaf else None, "stack": [fmt(f) for f in own[:8]]}
        s["count"] += 1
        if not s["iters_seen"] or s["iters_seen"][-1] != self.iter:
            s["iters_seen"].append(self.iter)

    def result(self, meta=None):
        n = max(self.iters, 1)
        sites = {k: dict(v, count_per_iter=v["count"] / n)
                 for k, v in sorted(self.sites.items(), key=lambda kv: -kv[1]["count"])}
        return {"meta": dict(meta or {}, iters_audited=self.iters),
                "total_syncs_per_iter": sum(v["count"] for v in self.sites.values()) / n,
                "sites": sites, "other_warnings": dict(self.other)}


# --------------------------------------------------------------------------------------------- profile summary
_DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
_RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}
_STEP_RE = re.compile(r"^ProfilerStep#\d+$")


def _union_ms(intervals, lo, hi):
    """Total length (ms) of the union of [start, end] µs intervals, clipped to [lo, hi]."""
    iv = sorted((max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi)
    total, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1000.0


def summarize_trace(events, prefix=PREFIX):
    """Per-step summary of a profiled window from chrome-trace events (the ``traceEvents`` list of
    ``prof.export_chrome_trace``; only complete ``"ph": "X"`` events with ``name``, ``cat``, ``ts``, ``dur`` µs
    and, for kernels and CUDA runtime calls, ``args.correlation`` are used).

    - steps: the ``ProfilerStep#N`` ranges; ``step_wall_ms`` is their mean duration, the window runs from the
      first step's start to the last step's end.
    - ``gpu_busy_fraction``: union of CUDA kernel intervals inside the window / window length (``_incl_memcpy``
      adds memcpy/memset; ``_excl_nccl`` drops NCCL kernels, whose time can be a wait for other ranks).
    - stages: ``audit/<name>`` CPU ranges (not the ``gpu_user_annotation`` copies). ``cpu_ms`` is the mean
      inclusive range time per step; ``cuda_ms`` is the summed duration of the device work *launched* inside
      the range (kernel matched to its launching runtime call by correlation id; kernels launched by the
      autograd thread during ``backward()`` fall inside the main thread's backward range by time). Nested
      stages are inclusive. ``head_convs`` (derived) = head - head_targets - head_loss.
    - host syncs: CPU time in ``cuda*Synchronize`` and blocking ``cudaMemcpy`` calls per step.
    """
    xs = [e for e in events if e.get("ph", "X") == "X" and "ts" in e and "dur" in e]
    steps = sorted((e for e in xs if _STEP_RE.match(str(e.get("name", "")))), key=lambda e: e["ts"])
    n = len(steps)
    if n == 0:
        return {"steps": 0, "error": "no ProfilerStep ranges in the trace"}
    lo = float(steps[0]["ts"])
    hi = max(float(e["ts"]) + float(e["dur"]) for e in steps)
    window_ms = (hi - lo) / 1000.0

    launch_ts = {}
    syncs = defaultdict(lambda: [0, 0.0])
    for e in xs:
        if e.get("cat") in _RUNTIME_CATS:
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                launch_ts[corr] = float(e["ts"])
            name = str(e.get("name", ""))
            if lo <= float(e["ts"]) <= hi and ("Synchronize" in name or name in ("cudaMemcpy", "cuMemcpyDtoH_v2")):
                syncs[name][0] += 1
                syncs[name][1] += float(e["dur"])

    dev = [e for e in xs if e.get("cat") in _DEVICE_CATS]
    kern = [e for e in dev if e.get("cat") == "kernel"]
    iv = lambda es: [(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in es]
    busy = _union_ms(iv(kern), lo, hi)
    busy_all = _union_ms(iv(dev), lo, hi)
    busy_no_nccl = _union_ms(iv([e for e in kern if "nccl" not in str(e.get("name", "")).lower()]), lo, hi)
    in_window = [e for e in kern if float(e["ts"]) < hi and float(e["ts"]) + float(e["dur"]) > lo]

    # device work by launch time, prefix sums for range queries
    launches = sorted(
        (launch_ts.get((e.get("args") or {}).get("correlation"), float(e["ts"])), float(e["dur"])) for e in dev)
    lts = [t for t, _ in launches]
    csum = [0.0]
    for _, d in launches:
        csum.append(csum[-1] + d)

    def dev_us(s, e):
        i, j = bisect.bisect_left(lts, s), bisect.bisect_right(lts, e)
        return csum[j] - csum[i]

    stages = defaultdict(lambda: {"cpu_us": 0.0, "cuda_us": 0.0, "calls": 0})
    for e in xs:
        name = str(e.get("name", ""))
        if not name.startswith(prefix) or e.get("cat") == "gpu_user_annotation":
            continue
        s, d = float(e["ts"]), float(e["dur"])
        if not (lo <= s <= hi):
            continue
        st = stages[name[len(prefix):]]
        st["cpu_us"] += d
        st["cuda_us"] += dev_us(s, s + d)
        st["calls"] += 1
    per_stage = {k: {"cpu_ms": v["cpu_us"] / n / 1000.0, "cuda_ms": v["cuda_us"] / n / 1000.0,
                     "calls_per_step": v["calls"] / n} for k, v in sorted(stages.items())}
    if {"head", "head_targets", "head_loss"} <= set(per_stage):
        per_stage["head_convs (derived)"] = {
            m: per_stage["head"][m] - per_stage["head_targets"][m] - per_stage["head_loss"][m]
            for m in ("cpu_ms", "cuda_ms")}

    # device work launched inside the window but outside every top-level stage range
    tops = sorted((float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in xs
                  if str(e.get("name", "")).startswith(prefix) and e.get("cat") != "gpu_user_annotation"
                  and lo <= float(e["ts"]) <= hi)
    merged = []
    for s, e in tops:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    staged_us = sum(dev_us(s, e) for s, e in merged)
    return {
        "steps": n,
        "step_wall_ms": sum(float(e["dur"]) for e in steps) / n / 1000.0,
        "window_ms": window_ms,
        "gpu_busy_ms_per_step": busy / n,
        "gpu_busy_fraction": busy / window_ms if window_ms else 0.0,
        "gpu_busy_fraction_incl_memcpy": busy_all / window_ms if window_ms else 0.0,
        "gpu_busy_fraction_excl_nccl": busy_no_nccl / window_ms if window_ms else 0.0,
        "per_step_busy_fraction": [
            _union_ms(iv(kern), float(e["ts"]), float(e["ts"]) + float(e["dur"])) / (float(e["dur"]) / 1000.0)
            if float(e["dur"]) > 0 else 0.0 for e in steps],
        "kernels_per_step": len(in_window) / n,
        "mean_kernel_us": (sum(float(e["dur"]) for e in in_window) / len(in_window)) if in_window else 0.0,
        "nccl_kernel_ms_per_step": sum(float(e["dur"]) for e in in_window
                                       if "nccl" in str(e.get("name", "")).lower()) / n / 1000.0,
        "host_sync": {k: {"calls_per_step": c / n, "ms_per_step": us / n / 1000.0} for k, (c, us) in syncs.items()},
        "device_ms_outside_stages_per_step": (dev_us(lo, hi) - staged_us) / n / 1000.0,
        "stages": per_stage,
    }


def summarize_trace_file(path):
    with open(path) as f:
        data = json.load(f)
    return summarize_trace(data["traceEvents"] if isinstance(data, dict) else data)


# --------------------------------------------------------------------------------------------- trainer hooks
class StepAudit:
    """Owned by the Trainer when an audit variable is set. Hooks: ``on_device_ready`` (after set_device),
    ``iter_start`` / ``iter_end`` (around every iteration; ``iter_end`` returns True when the run should stop) and
    ``finish`` (in ``after_train``)."""

    def __init__(self, rank: int, out_dir: str):
        global _MODE
        self.rank = rank
        self.iters = _env_int("YOLOX_AUDIT_ITERS", 0)
        self.sync_path = os.environ.get("YOLOX_SYNC_AUDIT", "").strip() or None
        self.prof_dir = os.environ.get("YOLOX_PROFILE", "").strip() or None
        cap = os.environ.get("YOLOX_MEM_CAP_GB", "").strip()
        self.mem_cap_gb = float(cap) if cap else None
        self.out_dir = os.environ.get("YOLOX_AUDIT_OUT", "").strip() or os.path.join(out_dir, "audit")
        if rank != 0:
            self.sync_path = self.prof_dir = None
        self.sync_warmup = _env_int("YOLOX_SYNC_AUDIT_WARMUP", 30)
        self.sync_steps = _env_int("YOLOX_SYNC_AUDIT_STEPS", 20)
        self.prof_wait, self.prof_warmup, self.prof_active = 30, 5, 10
        self.count = 0
        self.device = None
        self.prof = None
        self.summary = None
        self.sync = SyncRecorder() if self.sync_path else None
        self._sync_on = False
        self._sync_written = False
        _MODE = 2 if self.prof_dir else (1 if self.sync_path else 0)
        logger.info(f"step audit (rank {rank}): iters={self.iters or 'all'} sync={self.sync_path} "
                    f"profile={self.prof_dir} mem_cap_gb={self.mem_cap_gb} out={self.out_dir}")

    # -- device / memory
    def on_device_ready(self, local_rank: int):
        import torch

        if not torch.cuda.is_available():
            return
        self.device = torch.device("cuda", local_rank)
        if self.mem_cap_gb:
            total = torch.cuda.get_device_properties(self.device).total_memory
            frac = min(1.0, self.mem_cap_gb * 1024**3 / total)
            torch.cuda.set_per_process_memory_fraction(frac, self.device)
            logger.info(f"step audit: memory cap {self.mem_cap_gb} GiB of {total / 1024**3:.1f} GiB "
                        f"(fraction {frac:.4f}) on {self.device}")

    # -- per iteration
    def iter_start(self):
        if self.count == 0 and self.prof_dir:
            self._start_profiler()
        if self.sync is not None and not self._sync_written:
            if self.count == self.sync_warmup:
                self.sync.start()
                self._sync_on = True
            if self._sync_on:
                self.sync.next_iter()

    def iter_end(self, last_iter_of_epoch: bool = False) -> bool:
        self.count += 1
        if self._sync_on and self.count >= self.sync_warmup + self.sync_steps:
            self._stop_sync(complete=True)
        if self.prof is not None:
            self.prof.step()
        return bool(self.iters) and (self.count >= self.iters or last_iter_of_epoch)

    # -- profiler
    def _start_profiler(self):
        import torch
        from torch.profiler import ProfilerActivity, profile, schedule

        os.makedirs(self.prof_dir, exist_ok=True)
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
        self.prof = profile(activities=acts, schedule=schedule(wait=self.prof_wait, warmup=self.prof_warmup,
                                                               active=self.prof_active, repeat=1),
                            on_trace_ready=self._on_trace)
        self.prof.start()

    def _on_trace(self, prof):
        trace = os.path.join(self.prof_dir, f"trace_rank{self.rank}.json")
        prof.export_chrome_trace(trace)
        import torch

        sort = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        with open(os.path.join(self.prof_dir, "key_averages.txt"), "w") as f:
            f.write(prof.key_averages().table(sort_by=sort, row_limit=80))
        self.summary = summarize_trace_file(trace)
        self.summary["meta"] = self._meta(profile_schedule={"wait": self.prof_wait, "warmup": self.prof_warmup,
                                                            "active": self.prof_active})
        _write_json(os.path.join(self.prof_dir, "summary.json"), self.summary)
        logger.info(f"step audit: profile summary -> {self.prof_dir}/summary.json: "
                    f"step {self.summary.get('step_wall_ms', 0):.1f} ms, "
                    f"GPU busy {self.summary.get('gpu_busy_fraction', 0):.1%}")

    # -- sync
    def _stop_sync(self, complete):
        self.sync.stop()
        self._sync_on = False
        self._sync_written = True
        res = self.sync.result(self._meta(warmup=self.sync_warmup, planned_iters=self.sync_steps, complete=complete))
        _write_json(self.sync_path, res)
        logger.info(f"step audit: {res['total_syncs_per_iter']:.1f} host syncs/iter over {self.sync.iters} iters, "
                    f"{len(res['sites'])} sites -> {self.sync_path}")

    def _meta(self, **kw):
        import torch

        from yolox.utils import perf

        return dict(kw, rank=self.rank, torch=torch.__version__, perf=sorted(perf.flags()),
                    world_size=int(os.environ.get("WORLD_SIZE", "1") or 1), argv=sys.argv)

    # -- exit
    def finish(self, exc: BaseException | None = None):
        global _MODE
        if self._sync_on:
            self._stop_sync(complete=False)
        if self.prof is not None:
            try:
                self.prof.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"step audit: profiler stop failed: {e}")
            self.prof = None
            if self.summary is None:
                logger.warning(f"step audit: run ended after {self.count} iterations, before the profiler's active "
                               f"window ({self.prof_wait + self.prof_warmup + self.prof_active} iterations)")
        _MODE = 0
        _STACK.clear()
        self._write_mem(exc)

    def _write_mem(self, exc):
        import torch

        oom = exc is not None and ("out of memory" in str(exc).lower()
                                   or type(exc).__name__ == "OutOfMemoryError")
        rec = {"rank": self.rank, "iters_done": self.count, "mem_cap_gb": self.mem_cap_gb, "oom": oom,
               "error": None if exc is None else f"{type(exc).__name__}: {str(exc)[:500]}"}
        if self.device is not None:
            props = torch.cuda.get_device_properties(self.device)
            gib = 1024**3
            rec.update(device=str(self.device), device_name=props.name, total_gib=props.total_memory / gib,
                       max_allocated_gib=torch.cuda.max_memory_allocated(self.device) / gib,
                       max_reserved_gib=torch.cuda.max_memory_reserved(self.device) / gib)
            logger.info(f"step audit: rank {self.rank} peak allocated {rec['max_allocated_gib']:.2f} GiB, reserved "
                        f"{rec['max_reserved_gib']:.2f} GiB (cap {self.mem_cap_gb}), oom={oom}")
        _write_json(os.path.join(self.out_dir, f"mem_rank{self.rank}.json"), rec)


def _write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        logger.error("usage: python -m yolox.utils.step_audit <trace.json> [summary.json]")
        return 2
    res = summarize_trace_file(argv[0])
    out = argv[1] if len(argv) > 1 else os.path.splitext(argv[0])[0] + "_summary.json"
    _write_json(out, res)
    logger.info(f"wrote {out}: step {res.get('step_wall_ms', 0):.1f} ms, "
                f"GPU busy {res.get('gpu_busy_fraction', 0):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
