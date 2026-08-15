from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tinygrad import Tensor


@dataclass(frozen=True)
class WorkloadCase:
  name: str
  shape: tuple[int, ...]

  @property
  def key(self) -> str: return f"{self.name}:{','.join(map(str, self.shape))}"


def _values(shape: tuple[int, ...], scale: float = 97.0) -> Tensor:
  total = 1
  for value in shape: total *= value
  # Build from host storage. Tensor.arange is algebraically represented and can
  # remain device-neutral, which would make a later .to("VORTEX") ineffective.
  values = (np.arange(total, dtype=np.float32) % int(scale)) / np.float32(scale)
  return Tensor(values.reshape(shape), device="CPU").realize()


def build_case(case: WorkloadCase, device: str = "VORTEX", realize_inputs: bool = True) -> tuple[Tensor, Tensor]:
  """Return an unrealized device output and an already realized CPU reference."""
  if case.name == "vecadd":
    (n,) = case.shape
    a, b = _values((n,)), _values((n,), 89.0)
    reference = (a + b).realize()
    da, db = a.to(device), b.to(device)
    output = (da.realize() if realize_inputs else da) + (db.realize() if realize_inputs else db)
  elif case.name == "reduction":
    rows, columns = case.shape
    a = _values((rows, columns))
    reference = a.sum(axis=1).realize()
    da = a.to(device)
    output = (da.realize() if realize_inputs else da).sum(axis=1)
  elif case.name == "sgemm":
    m, n, k = case.shape
    a, b = _values((m, k)), _values((k, n), 89.0)
    reference = a.matmul(b).realize()
    da, db = a.to(device), b.to(device)
    output = (da.realize() if realize_inputs else da).matmul(db.realize() if realize_inputs else db)
  elif case.name == "conv":
    batch, channels_in, height, width, channels_out, kernel = case.shape
    x, weight = _values((batch, channels_in, height, width)), _values((channels_out, channels_in, kernel, kernel), 89.0)
    reference = x.conv2d(weight).realize()
    dx, dw = x.to(device), weight.to(device)
    output = (dx.realize() if realize_inputs else dx).conv2d(dw.realize() if realize_inputs else dw)
  else: raise ValueError(f"unknown workload {case.name!r}")
  return output, reference


def breadth_cases() -> tuple[WorkloadCase, ...]:
  cases = [WorkloadCase("vecadd", (n,)) for n in (256, 1024, 4096, 16384, 65536)]
  cases += [WorkloadCase("reduction", shape) for shape in ((1, 256), (16, 256), (64, 1024), (256, 1024), (64, 4096))]
  cases += [WorkloadCase("sgemm", (n, n, n)) for n in (32, 64, 128, 256)]
  cases += [WorkloadCase("conv", shape) for shape in ((1, 1, 28, 28, 8, 3), (1, 8, 13, 13, 16, 3),
                                                       (1, 4, 16, 16, 8, 3), (1, 8, 8, 8, 8, 3))]
  return tuple(cases)


def sgemm_depth_cases() -> tuple[WorkloadCase, ...]:
  shapes = ((32, 64, 128), (32, 128, 64), (64, 32, 128), (64, 128, 32), (128, 32, 64), (128, 64, 32),
            (64, 128, 256), (64, 256, 128), (128, 64, 256), (128, 256, 64), (256, 64, 128), (256, 128, 64))
  return tuple(WorkloadCase("sgemm", shape) for shape in shapes)


def all_cases(include_depth: bool = True) -> tuple[WorkloadCase, ...]:
  return breadth_cases() + (sgemm_depth_cases() if include_depth else ())


def parse_case(name: str, shape: str | Iterable[int]) -> WorkloadCase:
  dims = tuple(map(int, shape.split(","))) if isinstance(shape, str) else tuple(map(int, shape))
  expected = {"vecadd":1, "reduction":2, "sgemm":3, "conv":6}
  if name not in expected or len(dims) != expected[name]: raise ValueError(f"{name} expects {expected.get(name, 'a known number of')} dimensions")
  return WorkloadCase(name, dims)


class TinyMNISTConvNet:
  def __init__(self, channels: int = 8):
    from tinygrad import nn
    self.l1 = nn.Conv2d(1, channels, kernel_size=(3, 3))
    self.l2 = nn.Conv2d(channels, channels * 2, kernel_size=(3, 3))
    self.l3 = nn.Linear(channels * 2 * 5 * 5, 10)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2, 2))
    x = self.l2(x).relu().max_pool2d((2, 2))
    return self.l3(x.flatten(1))


def build_tinymnist(batch: int = 1, channels: int = 8, device: str = "VORTEX") -> tuple[Tensor, Tensor]:
  """Build deterministic TinyMNIST-shaped inference without downloading or training MNIST."""
  from tinygrad import nn
  from tinygrad.helpers import Context
  Tensor.manual_seed(42)
  with Context(DEV="CPU"): cpu_model = TinyMNISTConvNet(channels)
  x = _values((batch, 1, 28, 28), 255.0)
  for value in nn.state.get_state_dict(cpu_model).values(): value.realize()
  reference = cpu_model(x).realize()
  state = nn.state.get_state_dict(cpu_model)
  with Context(DEV=device): device_model = TinyMNISTConvNet(channels)
  nn.state.load_state_dict(device_model, {name:value.to(device) for name,value in state.items()}, verbose=False, realize=False)
  return device_model(x.to(device)), reference
