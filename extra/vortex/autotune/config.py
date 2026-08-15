from __future__ import annotations

import contextlib, hashlib, os
from dataclasses import asdict, dataclass
from typing import Iterator, Mapping


@dataclass(frozen=True, order=True)
class VortexHardwareConfig:
  """The performance-oriented Vortex design variables explored by the plugin."""

  threads: int
  warps: int
  cores: int
  clusters: int = 1

  def __post_init__(self):
    for name, value in asdict(self).items():
      if value <= 0 or value & (value - 1): raise ValueError(f"{name} must be a positive power of two, got {value}")

  @property
  def compute_lanes(self) -> int: return self.clusters * self.cores * self.threads

  @property
  def resident_warps(self) -> int: return self.clusters * self.cores * self.warps

  @property
  def resident_threads(self) -> int: return self.resident_warps * self.threads

  @property
  def local_size_limit(self) -> int: return self.threads * self.warps

  @property
  def key(self) -> str: return f"t{self.threads}-w{self.warps}-c{self.cores}-k{self.clusters}"

  def defines(self) -> str:
    return " ".join((f"-DVX_CFG_NUM_THREADS={self.threads}", f"-DVX_CFG_NUM_WARPS={self.warps}",
                     f"-DVX_CFG_NUM_CORES={self.cores}", f"-DVX_CFG_NUM_CLUSTERS={self.clusters}"))

  def to_dict(self) -> dict[str, int]: return asdict(self)

  @staticmethod
  def from_dict(data: Mapping[str, int]) -> VortexHardwareConfig:
    return VortexHardwareConfig(**{k:int(data[k]) for k in ("threads", "warps", "cores")}, clusters=int(data.get("clusters", 1)))

  @property
  def digest(self) -> str: return hashlib.sha256(self.defines().encode()).hexdigest()[:12]

  @contextlib.contextmanager
  def environment(self, build_runtime: bool = False) -> Iterator[None]:
    """Temporarily expose this configuration to the existing Vortex backend."""
    names = ("VORTEX_CONFIGS", "VORTEX_BUILD_RUNTIME", "VORTEX_PROFILING")
    old = {name:os.environ.get(name) for name in names}
    os.environ["VORTEX_CONFIGS"] = self.defines()
    os.environ["VORTEX_PROFILING"] = "1"
    if build_runtime: os.environ["VORTEX_BUILD_RUNTIME"] = "1"
    try: yield
    finally:
      for name, value in old.items():
        if value is None: os.environ.pop(name, None)
        else: os.environ[name] = value


def default_hardware_grid() -> tuple[VortexHardwareConfig, ...]:
  return tuple(VortexHardwareConfig(t, w, c) for t in (4, 8, 16, 32) for w in (2, 4, 8) for c in (1, 2, 4))

