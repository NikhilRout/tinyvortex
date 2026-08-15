from __future__ import annotations

import hashlib, random
from dataclasses import dataclass
from typing import Iterable, Sequence

from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.search import get_kernel_actions
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import AxisType, UOp


@dataclass(frozen=True)
class ScheduleCandidate:
  opts: tuple[Opt, ...]
  origin: str = "search"

  @property
  def key(self) -> tuple[Opt, ...]: return self.opts


def base_scheduler(ast: UOp, renderer: Renderer) -> Scheduler:
  scheduler = Scheduler(ast, renderer)
  scheduler.convert_loop_to_global()
  return scheduler


def replay_schedule(ast: UOp, renderer: Renderer, opts: Sequence[Opt]) -> Scheduler:
  scheduler = base_scheduler(ast, renderer)
  for opt in opts: scheduler.apply_opt(opt)
  return scheduler


def schedule_local_size(scheduler: Scheduler) -> int:
  size = 1
  for extent, axis_type in zip(scheduler.full_shape, scheduler.axis_types):
    if axis_type not in (AxisType.LOCAL, AxisType.GROUP_REDUCE, AxisType.THREAD): continue
    size *= int(extent) if isinstance(extent, int) else int(extent.vmax)
  return size


def _add(out: list[ScheduleCandidate], seen: set[tuple[Opt, ...]], scheduler: Scheduler, origin: str) -> None:
  opts = tuple(scheduler.applied_opts)
  if opts not in seen:
    seen.add(opts)
    out.append(ScheduleCandidate(opts, origin))


def enumerate_vecadd_schedules(ast: UOp, renderer: Renderer, limit: int = 64,
                               upcasts: Sequence[int] = (1, 2, 4, 8, 16)) -> list[ScheduleCandidate]:
  """Enumerate the complete useful 1-D VecAdd UPCAST x LOCAL product.

  A factor of one means that the corresponding transformation is omitted.
  Legality is determined by Scheduler itself, and LOCAL is bounded by the
  renderer's hardware-specific local-size limit.
  """
  if limit < 2: raise ValueError("schedule limit must be at least two")
  local_limit = int(renderer.local_max[0]) if renderer.local_max is not None else 1
  locals_ = tuple(1 << power for power in range(local_limit.bit_length()) if 1 << power <= local_limit)
  default_opts: tuple[Opt, ...] = ()
  try: default_opts = tuple(hand_coded_optimizations(base_scheduler(ast, renderer).copy()).applied_opts)
  except (KernelOptError, AssertionError, RuntimeError, ValueError): pass
  out, seen = [], set()
  for upcast in upcasts:
    for local in locals_:
      scheduler = base_scheduler(ast, renderer)
      try:
        if upcast > 1: scheduler.apply_opt(Opt(OptOps.UPCAST, 0, upcast))
        if local > 1: scheduler.apply_opt(Opt(OptOps.LOCAL, 0, local))
      except (KernelOptError, AssertionError, RuntimeError, ValueError): continue
      opts = tuple(scheduler.applied_opts)
      origin = "tinygrad-default" if opts == default_opts else f"vecadd-u{upcast}-l{local}"
      _add(out, seen, scheduler, origin)
  return out[:limit]


def enumerate_sgemm_schedules(ast: UOp, renderer: Renderer, limit: int = 160) -> list[ScheduleCandidate]:
  """Deterministic structured sample of square SGEMM output/reduction tilings.

  The canonical order matches tinygrad's heuristic: N upcast, M upcast,
  K unroll, N local, then M local. The complete legal universe is much
  larger than a practical SimX corpus, so the default and unoptimized points
  are retained and the rest are sampled reproducibly across the full grid.
  """
  if limit < 2: raise ValueError("schedule limit must be at least two")
  local_limit = int(renderer.local_max[0]) if renderer.local_max is not None else 1
  local_factors = tuple(1 << power for power in range(local_limit.bit_length()) if 1 << power <= local_limit)
  output_tiles = tuple((um, un) for um in (1, 2, 4, 8) for un in (1, 2, 4, 8) if um * un <= 16)
  default_opts: tuple[Opt, ...] = ()
  try: default_opts = tuple(hand_coded_optimizations(base_scheduler(ast, renderer).copy()).applied_opts)
  except (KernelOptError, AssertionError, RuntimeError, ValueError): pass
  candidates: list[ScheduleCandidate] = []
  seen: set[tuple[Opt, ...]] = set()
  for um, un in output_tiles:
    for uk in (1, 2, 4, 8, 16):
      for lm in local_factors:
        for ln in local_factors:
          if lm * ln > local_limit: continue
          scheduler = base_scheduler(ast, renderer)
          try:
            if un > 1: scheduler.apply_opt(Opt(OptOps.UPCAST, 1, un))
            if um > 1: scheduler.apply_opt(Opt(OptOps.UPCAST, 0, um))
            if uk > 1: scheduler.apply_opt(Opt(OptOps.UNROLL, 0, uk))
            if ln > 1: scheduler.apply_opt(Opt(OptOps.LOCAL, 1, ln))
            if lm > 1: scheduler.apply_opt(Opt(OptOps.LOCAL, 0, lm))
          except (KernelOptError, AssertionError, RuntimeError, ValueError): continue
          opts = tuple(scheduler.applied_opts)
          if opts in seen: continue
          seen.add(opts)
          origin = "tinygrad-default" if opts == default_opts else f"sgemm-um{um}-un{un}-uk{uk}-lm{lm}-ln{ln}"
          candidates.append(ScheduleCandidate(opts, origin))
  # Small SGEMMs can favor cooperative reduction rather than register tiling.
  group_candidates: list[ScheduleCandidate] = []
  for kind in (OptOps.GROUP, OptOps.GROUPTOP):
    for amount in (2, 4, 8, 16):
      scheduler = base_scheduler(ast, renderer)
      try: scheduler.apply_opt(Opt(kind, 0, amount))
      except (KernelOptError, AssertionError, RuntimeError, ValueError): continue
      opts = tuple(scheduler.applied_opts)
      if opts in seen: continue
      seen.add(opts)
      origin = "tinygrad-default" if opts == default_opts else f"sgemm-{kind.name.lower()}{amount}"
      candidate = ScheduleCandidate(opts, origin)
      candidates.append(candidate)
      group_candidates.append(candidate)
  priority = [ScheduleCandidate(default_opts, "tinygrad-default")]
  if default_opts: priority.append(ScheduleCandidate((), "unoptimized"))
  priority.extend(candidate for candidate in group_candidates if candidate.opts not in {x.opts for x in priority})
  selected, selected_keys = list(priority), {candidate.opts for candidate in priority}
  remaining = [candidate for candidate in candidates if candidate.opts not in selected_keys]
  remaining.sort(key=lambda candidate:hashlib.sha256(ast.key + repr(candidate.opts).encode()).digest())
  selected.extend(remaining[:max(0, limit - len(selected))])
  return selected[:limit]


def enumerate_schedules(ast: UOp, renderer: Renderer, limit: int = 24, max_depth: int = 6, seed: int | None = None,
                        schedule_space: str = "generic") -> list[ScheduleCandidate]:
  if limit < 2: raise ValueError("schedule limit must be at least two")
  if schedule_space == "vecadd": return enumerate_vecadd_schedules(ast, renderer, limit)
  if schedule_space == "sgemm": return enumerate_sgemm_schedules(ast, renderer, limit)
  if schedule_space != "generic": raise ValueError(f"unknown schedule space {schedule_space!r}")
  base = base_scheduler(ast, renderer)
  out: list[ScheduleCandidate] = [ScheduleCandidate((), "unoptimized")]
  seen: set[tuple[Opt, ...]] = {()}

  try:
    default = hand_coded_optimizations(base.copy())
    _add(out, seen, default, "tinygrad-default")
  except (KernelOptError, AssertionError, RuntimeError, ValueError): pass

  actions = list(get_kernel_actions(base, include_0=False).values())
  for kind in OptOps:
    if candidate := next((x for x in actions if x.applied_opts and x.applied_opts[-1].op is kind), None):
      _add(out, seen, candidate, f"cover-{kind.name.lower()}")
    if len(out) >= limit: return out[:limit]

  if seed is None: seed = int.from_bytes(hashlib.sha256(ast.key).digest()[:8], "little")
  rng, frontier = random.Random(seed), actions
  rng.shuffle(frontier)
  depth = 1
  while frontier and len(out) < limit and depth <= max_depth:
    next_frontier: list[Scheduler] = []
    for state in frontier:
      _add(out, seen, state, "search")
      if len(out) >= limit: break
      children = list(get_kernel_actions(state, include_0=False).values())
      rng.shuffle(children)
      next_frontier.extend(children[:max(2, limit // 4)])
    rng.shuffle(next_frontier)
    frontier, depth = next_frontier[:limit * 2], depth + 1
  return out[:limit]


def serialize_opts(opts: Iterable[Opt]) -> list[dict]:
  return [{"op":x.op.name, "axis":x.axis, "arg":list(x.arg) if isinstance(x.arg, tuple) else x.arg} for x in opts]


def deserialize_opts(data: Iterable[dict]) -> tuple[Opt, ...]:
  return tuple(Opt(OptOps[x["op"]], x.get("axis"), tuple(x["arg"]) if isinstance(x.get("arg"), list) else x.get("arg")) for x in data)
