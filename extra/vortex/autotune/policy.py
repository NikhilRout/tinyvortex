from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence

from tinygrad.codegen.opt import Opt
from tinygrad.device import Device
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.vortex.autotune.config import VortexHardwareConfig
from extra.vortex.autotune.features import extract_features
from extra.vortex.autotune.model import MODEL_VERSION, CycleModel
from extra.vortex.autotune.search import enumerate_schedules, replay_schedule, serialize_opts

RendererFactory = Callable[[VortexHardwareConfig], Renderer]


def _is_vortex_call(call: UOp) -> bool:
  devices = (call.device,) if isinstance(call.device, str) else tuple(call.device) if call.device is not None else ()
  return call.op is Ops.CALL and "VORTEX" in devices


@dataclass(frozen=True)
class ScoredDesign:
  config: VortexHardwareConfig
  schedule: tuple[Opt, ...]
  predicted_cycles: float
  confidence: float
  origin: str = "search"

  def to_dict(self) -> dict:
    return {"hardware":self.config.to_dict(), "schedule":serialize_opts(self.schedule), "predicted_cycles":self.predicted_cycles,
            "confidence":self.confidence, "origin":self.origin}


@dataclass(frozen=True)
class DesignRecommendation:
  selected: ScoredDesign
  runners_up: tuple[ScoredDesign, ...] = ()

  def to_dict(self) -> dict: return {"selected":self.selected.to_dict(), "runners_up":[x.to_dict() for x in self.runners_up]}


@dataclass(frozen=True)
class KernelRecommendation:
  call_index: int
  ast_key: str
  schedule: tuple[Opt, ...]
  predicted_cycles: float
  confidence: float

  def to_dict(self) -> dict:
    return {"call_index":self.call_index, "ast_key":self.ast_key, "schedule":serialize_opts(self.schedule),
            "predicted_cycles":self.predicted_cycles, "confidence":self.confidence}


@dataclass(frozen=True)
class ApplicationRecommendation:
  config: VortexHardwareConfig
  kernels: tuple[KernelRecommendation, ...]
  predicted_cycles: float
  confidence: float
  runners_up: tuple[tuple[VortexHardwareConfig, float], ...] = ()

  def to_dict(self) -> dict:
    return {"hardware":self.config.to_dict(), "kernels":[x.to_dict() for x in self.kernels], "predicted_cycles":self.predicted_cycles,
            "confidence":self.confidence, "runners_up":[{"hardware":c.to_dict(), "predicted_cycles":v} for c,v in self.runners_up]}


def _default_renderer_factory(config: VortexHardwareConfig) -> Renderer:
  from tinygrad.runtime.ops_vortex import VortexRenderer
  with config.environment(): return VortexRenderer(Target(device="VORTEX", interface="SIMX"))


class VortexAutoScheduler:
  def __init__(self, model: CycleModel | str, renderer_factory: RendererFactory | None = None,
               schedule_limit: int = 24, max_depth: int = 6, schedule_space: str = "generic"):
    self.model = CycleModel.load(model) if isinstance(model, str) else model
    self.renderer_factory = renderer_factory or _default_renderer_factory
    self.schedule_limit, self.max_depth, self.schedule_space = schedule_limit, max_depth, schedule_space
    self._cache: dict[tuple[int, bytes, tuple[tuple[str, int], ...], str], ScoredDesign] = {}

  @staticmethod
  def _cache_key(ast: UOp, config: VortexHardwareConfig, var_vals: Mapping[str, int] | None) -> tuple[int, bytes, tuple[tuple[str, int], ...], str]:
    return MODEL_VERSION, ast.key, tuple(sorted((str(key), int(value)) for key, value in (var_vals or {}).items())), config.key

  def score(self, ast: UOp, schedule: Sequence[Opt], config: VortexHardwareConfig,
            var_vals: Mapping[str, int] | None = None, renderer: Renderer | None = None, origin: str = "manual") -> ScoredDesign:
    ren = renderer or self.renderer_factory(config)
    scheduler = replay_schedule(ast, ren, schedule)
    cycles, confidence = self.model.predict(extract_features(ast, scheduler, schedule, config, var_vals))
    return ScoredDesign(config, tuple(schedule), cycles, confidence, origin)

  def _select_schedule(self, ast: UOp, config: VortexHardwareConfig, var_vals: Mapping[str, int] | None,
                       renderer: Renderer) -> tuple[ScoredDesign, list[ScoredDesign]]:
    key = self._cache_key(ast, config, var_vals)
    if key in self._cache: return self._cache[key], [self._cache[key]]
    candidates = enumerate_schedules(ast, renderer, self.schedule_limit, self.max_depth, schedule_space=self.schedule_space)
    scored = sorted((self.score(ast, x.opts, config, var_vals, renderer, x.origin) for x in candidates), key=lambda x:x.predicted_cycles)
    self._cache[key] = scored[0]
    return scored[0], scored

  def select_schedule(self, ast: UOp, config: VortexHardwareConfig, var_vals: Mapping[str, int] | None = None) -> ScoredDesign:
    key = self._cache_key(ast, config, var_vals)
    if key in self._cache: return self._cache[key]
    return self._select_schedule(ast, config, var_vals, self.renderer_factory(config))[0]

  def recommend(self, ast: UOp, configs: Iterable[VortexHardwareConfig], top_k: int = 5,
                var_vals: Mapping[str, int] | None = None) -> DesignRecommendation:
    if top_k < 1: raise ValueError("top_k must be positive")
    designs: list[ScoredDesign] = []
    for config in configs:
      selected, _ = self._select_schedule(ast, config, var_vals, self.renderer_factory(config))
      designs.append(selected)
    if not designs: raise ValueError("at least one hardware configuration is required")
    designs.sort(key=lambda x:x.predicted_cycles)
    return DesignRecommendation(designs[0], tuple(designs[1:top_k]))

  def recommend_application(self, linear: UOp, configs: Iterable[VortexHardwareConfig], top_k: int = 5,
                            var_vals: Mapping[str, int] | None = None) -> ApplicationRecommendation:
    if linear.op is not Ops.LINEAR: raise ValueError("recommend_application expects a LINEAR UOp")
    per_config: list[tuple[VortexHardwareConfig, tuple[KernelRecommendation, ...], float, float]] = []
    for config in configs:
      renderer, kernels = self.renderer_factory(config), []
      for call_index, call in enumerate(linear.src):
        if not _is_vortex_call(call) or not call.src or call.src[0].op is not Ops.SINK: continue
        design, _ = self._select_schedule(call.src[0], config, var_vals, renderer)
        kernels.append(KernelRecommendation(call_index, call.src[0].key.hex(), design.schedule, design.predicted_cycles, design.confidence))
      if kernels:
        total = sum(x.predicted_cycles for x in kernels)
        confidence = math.exp(sum(math.log(max(x.confidence, 1e-12)) for x in kernels) / len(kernels))
        per_config.append((config, tuple(kernels), total, confidence))
    if not per_config: raise ValueError("linear schedule contains no compilable kernel calls")
    per_config.sort(key=lambda x:x[2])
    selected = per_config[0]
    return ApplicationRecommendation(selected[0], selected[1], selected[2], selected[3],
                                     tuple((x[0], x[2]) for x in per_config[1:top_k]))


def rewrite_linear(linear: UOp, recommendation: ApplicationRecommendation | Sequence[KernelRecommendation]) -> UOp:
  kernels = recommendation.kernels if isinstance(recommendation, ApplicationRecommendation) else tuple(recommendation)
  by_index = {x.call_index:x for x in kernels}
  calls = []
  for index, call in enumerate(linear.src):
    if index not in by_index:
      calls.append(call)
      continue
    if call.op is not Ops.CALL or not call.src or call.src[0].op is not Ops.SINK: raise ValueError(f"call {index} is not a schedulable kernel")
    rec, ast = by_index[index], call.src[0]
    if ast.key.hex() != rec.ast_key: raise ValueError(f"AST mismatch for call {index}")
    info = ast.arg if isinstance(ast.arg, KernelInfo) else KernelInfo()
    rewritten = ast.replace(arg=replace(info, opts_to_apply=rec.schedule))
    calls.append(call.replace(src=(rewritten, *call.src[1:])))
  missing = set(by_index) - set(range(len(linear.src)))
  if missing: raise ValueError(f"recommendations refer to missing calls: {sorted(missing)}")
  return linear.replace(src=tuple(calls))


def realize_with_policy(*tensors, recommendation: ApplicationRecommendation, update_stats: bool = True):
  """Realize tensors using one recommended Vortex configuration and its per-kernel schedules."""
  if not tensors: raise ValueError("at least one Tensor is required")
  linear, var_vals = tensors[0].linear_with_vars(*tensors[1:])
  linear = rewrite_linear(linear, recommendation)
  if "VORTEX" in Device._opened_devices:
    active = Device["VORTEX"].config.configs
    if active != recommendation.config.defines():
      raise RuntimeError("VORTEX is already open with a different configuration; run each hardware recommendation in a fresh process")
  with recommendation.config.environment(build_runtime=True): run_linear(linear, var_vals, update_stats=update_stats)
  return tensors[0] if len(tensors) == 1 else tensors
