from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence

from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import AxisType, UOp, sym_infer

from extra.vortex.autotune.config import VortexHardwareConfig

MAX_RANGES, MAX_SCHEDULE_OPS = 8, 6
TRACKED_UOPS = (
  "PARAM", "INDEX", "LOAD", "STORE", "CAST", "BITCAST", "EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL", "NEG", "ADD", "MUL",
  "MAX", "CMPLT", "CMPNE", "CMPEQ", "WHERE", "MULACC", "BARRIER", "RANGE", "IF", "END", "STAGE", "REDUCE", "WMMA", "CONST",
)
OPT_KINDS = tuple(OptOps)
AXIS_KINDS = tuple(AxisType)


def _log2p(value: float) -> float: return math.log2(max(0.0, value) + 1.0)


def _value(value, var_vals: Mapping[str, int]) -> float:
  try: return float(sym_infer(value, dict(var_vals)))
  except (TypeError, ValueError, KeyError, AttributeError):
    try: return float(value)
    except (TypeError, ValueError): return float(getattr(value, "vmax", 0))


def _depths(uops: Sequence[UOp]) -> tuple[int, float]:
  depths: dict[UOp, int] = {}
  for uop in uops: depths[uop] = 1 + max((depths.get(src, 0) for src in uop.src), default=0)
  return max(depths.values(), default=0), sum(depths.values()) / max(1, len(depths))


def _schedule_arg(arg) -> float:
  if isinstance(arg, tuple): return sum((i + 1) * _schedule_arg(x) for i, x in enumerate(arg))
  return float(arg or 0)


def feature_names() -> tuple[str, ...]:
  names = ["uops", "dag_depth", "dag_mean_depth", "edges", "max_fanin", "mean_fanout", "params", "buffers", "dtype_bytes_max",
           "estimate_ops", "estimate_load_store_bytes", "estimate_unique_bytes", "arithmetic_intensity"]
  names += [f"uop_{name.lower()}" for name in TRACKED_UOPS]
  names += [f"axis_count_{axis.name.lower()}" for axis in AXIS_KINDS]
  for i in range(MAX_RANGES): names += [f"range_{i}_extent", f"range_{i}_axis"]
  names += ["schedule_length"]
  for i in range(MAX_SCHEDULE_OPS):
    names += [f"schedule_{i}_{op.name.lower()}" for op in OPT_KINDS]
    names += [f"schedule_{i}_axis", f"schedule_{i}_arg"]
  names += ["hw_threads", "hw_warps", "hw_cores", "hw_clusters", "hw_compute_lanes", "hw_resident_warps",
            "hw_resident_threads", "hw_local_size_limit"]
  return tuple(names)


FEATURE_NAMES = feature_names()


def extract_features(ast: UOp, scheduler: Scheduler, schedule: Sequence[Opt], config: VortexHardwareConfig,
                     var_vals: Mapping[str, int] | None = None) -> list[float]:
  """Extract a stable, versioned numeric representation of a complete HW/SW design point."""
  vals = var_vals or {}
  uops = ast.toposort()
  counts = Counter(u.op.name for u in uops)
  depth, mean_depth = _depths(uops)
  edges = sum(len(u.src) for u in uops)
  fanouts = Counter(src for u in uops for src in u.src)
  max_fanin = max((len(u.src) for u in uops), default=0)
  dtype_bytes = max((u.dtype.scalar().itemsize for u in uops), default=0)
  try: estimates = Estimates.from_uops(tuple(uops))
  except (AssertionError, RuntimeError, ValueError): estimates = Estimates()
  est_ops, est_lds, est_mem = (_value(x, vals) for x in (estimates.ops, estimates.lds, estimates.mem))
  features = [
    _log2p(len(uops)), _log2p(depth), _log2p(mean_depth), _log2p(edges), _log2p(max_fanin),
    _log2p(sum(fanouts.values()) / max(1, len(fanouts))), _log2p(counts["PARAM"]), _log2p(counts["BUFFER"]), _log2p(dtype_bytes),
    _log2p(est_ops), _log2p(est_lds), _log2p(est_mem), math.log2((est_ops + 1.0) / (est_mem + 1.0)),
  ]
  features += [_log2p(counts[name]) for name in TRACKED_UOPS]
  axis_counts = Counter(scheduler.axis_types)
  features += [_log2p(axis_counts[axis]) for axis in AXIS_KINDS]
  ranges = list(zip(scheduler.full_shape, scheduler.axis_types))[:MAX_RANGES]
  for i in range(MAX_RANGES):
    if i < len(ranges): features += [_log2p(_value(ranges[i][0], vals)), float(AXIS_KINDS.index(ranges[i][1]) + 1) / len(AXIS_KINDS)]
    else: features += [0.0, 0.0]
  features.append(float(len(schedule)) / MAX_SCHEDULE_OPS)
  for i in range(MAX_SCHEDULE_OPS):
    opt = schedule[i] if i < len(schedule) else None
    features += [1.0 if opt is not None and opt.op is kind else 0.0 for kind in OPT_KINDS]
    features += [0.0 if opt is None or opt.axis is None else (opt.axis + 1.0) / MAX_RANGES,
                 0.0 if opt is None else _log2p(abs(_schedule_arg(opt.arg))) / 16.0]
  features += [_log2p(config.threads), _log2p(config.warps), _log2p(config.cores), _log2p(config.clusters),
               _log2p(config.compute_lanes), _log2p(config.resident_warps), _log2p(config.resident_threads),
               _log2p(config.local_size_limit)]
  assert len(features) == len(FEATURE_NAMES)
  return features
