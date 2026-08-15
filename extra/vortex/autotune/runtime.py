from __future__ import annotations

import ctypes, hashlib, time
from dataclasses import dataclass
from typing import Mapping, Sequence

from tinygrad.device import Device
from tinygrad.engine.realize import ExecContext, compile_linear, link_linear, pm_exec
from tinygrad.uop.ops import Ops, UOp

from extra.vortex.autotune.config import VortexHardwareConfig

VX_CSR_MCYCLE, VX_CSR_MINSTRET = 0xB00, 0xB02


@dataclass(frozen=True)
class PerformanceCounters:
  cycles_per_core: tuple[int, ...]
  instructions_per_core: tuple[int, ...]

  @property
  def cycles(self) -> int: return max(self.cycles_per_core, default=0)

  @property
  def instructions(self) -> int: return sum(self.instructions_per_core)

  def delta(self, previous: PerformanceCounters) -> PerformanceCounters:
    # Some drivers reset counters at launch; treat a decreasing value as a fresh per-kernel counter.
    cycles = tuple(after - before if after >= before else after for before, after in zip(previous.cycles_per_core, self.cycles_per_core))
    instrs = tuple(after - before if after >= before else after for before, after in zip(previous.instructions_per_core, self.instructions_per_core))
    return PerformanceCounters(cycles, instrs)


@dataclass(frozen=True)
class MeasuredExecution:
  per_call: dict[int, PerformanceCounters]
  compile_seconds: float
  source_hashes: dict[int, str]

  @property
  def total_cycles(self) -> int: return sum(x.cycles for x in self.per_call.values())

  @property
  def total_instructions(self) -> int: return sum(x.instructions for x in self.per_call.values())


def validate_runtime_config(expected: VortexHardwareConfig) -> dict[str, int]:
  """Fail before measurement if SimX does not match the design-point label."""
  caps = Device["VORTEX"].caps()
  mismatches = {name:(getattr(expected, name), int(caps[name])) for name in ("threads", "warps", "cores", "clusters")
                if int(caps[name]) != getattr(expected, name)}
  if mismatches:
    detail = ", ".join(f"{name}=expected {want}, runtime {got}" for name, (want, got) in mismatches.items())
    raise RuntimeError(f"Vortex runtime configuration mismatch: {detail}")
  return caps


def read_performance_counters() -> PerformanceCounters:
  device = Device["VORTEX"]
  lib, rt = device.rt.lib, device.rt
  lib.vx_mpm_query.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
  lib.vx_mpm_query.restype = ctypes.c_int
  cores = int(device.caps()["cores"] * device.caps()["clusters"])

  def read(addr: int, core: int) -> int:
    value = ctypes.c_uint64()
    rt.check(lib.vx_mpm_query(rt.dev, 0, addr, core, ctypes.byref(value)), "vx_mpm_query")
    return int(value.value)
  return PerformanceCounters(tuple(read(VX_CSR_MCYCLE, i) for i in range(cores)), tuple(read(VX_CSR_MINSTRET, i) for i in range(cores)))


def execute_linear_measured(linear: UOp, var_vals: Mapping[str, int] | None = None,
                            measured_calls: Sequence[int] | None = None) -> MeasuredExecution:
  """Compile a LINEAR graph once, execute it in order, and collect counter deltas around selected Vortex kernels."""
  started = time.perf_counter()
  compiled = link_linear(compile_linear(linear, beam=0))
  compile_seconds = time.perf_counter() - started
  selected = set(measured_calls if measured_calls is not None else range(len(compiled.src)))
  ctx, measurements, hashes = ExecContext(dict(var_vals or {}), update_stats=False, jit=True, wait=True, cache=False), {}, {}
  for index, call in enumerate(compiled.src):
    devices = (call.device,) if isinstance(call.device, str) else tuple(call.device) if call.device is not None else ()
    is_vortex_kernel = call.op is Ops.CALL and "VORTEX" in devices and call.src and call.src[0].op is Ops.PROGRAM and index in selected
    before = read_performance_counters() if is_vortex_kernel else None
    pm_exec.rewrite(call, ctx)
    if before is not None:
      measurements[index] = read_performance_counters().delta(before)
      program = call.src[0]
      source = next((x.arg for x in program.src if x.op is Ops.SOURCE), "")
      hashes[index] = hashlib.sha256(str(source).encode()).hexdigest()
  return MeasuredExecution(measurements, compile_seconds, hashes)
