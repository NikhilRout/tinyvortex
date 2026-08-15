from __future__ import annotations

import json, math, pathlib, random
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from tinygrad import Tensor, nn
from tinygrad.helpers import Context
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict

from extra.vortex.autotune.features import FEATURE_NAMES

MODEL_VERSION = 1


class _MLP:
  def __init__(self, input_dim: int):
    self.layers = [nn.Linear(input_dim, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 1)]

  def __call__(self, x: Tensor) -> Tensor:
    for layer in self.layers[:-1]: x = layer(x).gelu()
    return self.layers[-1](x).squeeze(-1)


@dataclass(frozen=True)
class TrainConfig:
  epochs: int = 100
  batch_size: int = 256
  learning_rate: float = 1e-3
  pairwise_weight: float = 0.25
  huber_delta: float = 1.0
  seed: int = 42
  hardware_agnostic: bool = False


class CycleModel:
  """Structured MLP and its feature/target calibration state."""

  def __init__(self, feature_mean: Sequence[float] | None = None, feature_std: Sequence[float] | None = None,
               target_mean: float = 0.0, target_std: float = 1.0, residual_rmse: float = 1.0,
               masked_features: Sequence[int] = ()):
    # Never inherit DEV=SIMX+VORTEX from a collection shell.  The predictor and
    # optimizer are deliberately host-side; Vortex is only the label oracle.
    with Context(DEV="CPU"):
      self.network = _MLP(len(FEATURE_NAMES))
    self.feature_mean = np.asarray(feature_mean if feature_mean is not None else np.zeros(len(FEATURE_NAMES)), dtype=np.float32)
    self.feature_std = np.asarray(feature_std if feature_std is not None else np.ones(len(FEATURE_NAMES)), dtype=np.float32)
    self.target_mean, self.target_std, self.residual_rmse = float(target_mean), float(target_std), float(residual_rmse)
    self.masked_features = tuple(map(int, masked_features))

  def normalize(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).copy()
    if self.masked_features: values[..., self.masked_features] = 0.0
    return (values - self.feature_mean) / self.feature_std

  def predict_log_cycles(self, features: Sequence[float]) -> float:
    x = Tensor(self.normalize(features).reshape(1, -1), device="CPU")
    normalized = float(self.network(x).item())
    return normalized * self.target_std + self.target_mean

  def predict(self, features: Sequence[float]) -> tuple[float, float]:
    normalized = self.normalize(features)
    log_cycles = self.predict_log_cycles(features)
    novelty = float(np.sqrt(np.mean(np.square(np.clip(normalized, -8.0, 8.0)))))
    uncertainty = math.exp(min(4.0, self.residual_rmse * (1.0 + 0.05 * novelty)))
    return math.exp(log_cycles), 1.0 / uncertainty

  def save(self, path: str | pathlib.Path, metadata: Mapping | None = None) -> None:
    output = pathlib.Path(path)
    if output.suffix != ".npz": raise ValueError("model artifact must use the .npz suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {name:t.numpy() for name, t in get_state_dict(self.network).items()}
    np.savez(output, **state)
    payload = {
      "model_version": MODEL_VERSION, "feature_names": FEATURE_NAMES, "feature_mean":self.feature_mean.tolist(),
      "feature_std":self.feature_std.tolist(), "target_mean":self.target_mean, "target_std":self.target_std,
      "residual_rmse":self.residual_rmse, "masked_features":self.masked_features, **dict(metadata or {}),
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

  @staticmethod
  def load(path: str | pathlib.Path) -> CycleModel:
    source = pathlib.Path(path)
    metadata = json.loads(source.with_suffix(".json").read_text())
    if metadata.get("model_version") != MODEL_VERSION: raise ValueError(f"unsupported model version {metadata.get('model_version')}")
    if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES: raise ValueError("model feature schema does not match this checkout")
    model = CycleModel(metadata["feature_mean"], metadata["feature_std"], metadata["target_mean"], metadata["target_std"],
                       metadata["residual_rmse"], metadata.get("masked_features", ()))
    with np.load(source, allow_pickle=False) as values:
      load_state_dict(model.network, {name:Tensor(values[name].copy(), device="CPU") for name in values.files}, strict=True, verbose=False)
    return model


def _huber(diff: Tensor, delta: float) -> Tensor:
  absolute = diff.abs()
  return (absolute <= delta).where(0.5 * diff.square(), delta * (absolute - 0.5 * delta)).mean()


def _pairs(records: Sequence[Mapping], limit_per_group: int = 64) -> list[tuple[int, int]]:
  groups: dict[str, list[int]] = {}
  for i, record in enumerate(records): groups.setdefault(str(record.get("group_key", record.get("workload", "default"))), []).append(i)
  pairs: list[tuple[int, int]] = []
  for indexes in groups.values():
    ordered = sorted(indexes, key=lambda i:float(records[i]["cycles"]))
    if len(ordered) < 2: continue
    candidates = [(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered))]
    pairs.extend(candidates[:limit_per_group])
  return pairs


def train_cycle_model(records: Iterable[Mapping], output: str | pathlib.Path, config: TrainConfig = TrainConfig()) -> CycleModel:
  rows = [record for record in records if record.get("correct", True) and float(record.get("cycles", 0)) > 0]
  if len(rows) < 2: raise ValueError("at least two correct cycle measurements are required")
  x = np.asarray([row["features"] for row in rows], dtype=np.float32)
  if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES): raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {x.shape}")
  masked_features = tuple(i for i, name in enumerate(FEATURE_NAMES) if name.startswith("hw_")) if config.hardware_agnostic else ()
  if masked_features: x[:, masked_features] = 0.0
  y_log = np.log(np.asarray([float(row["cycles"]) for row in rows], dtype=np.float32))
  feature_mean, feature_std = x.mean(axis=0), x.std(axis=0)
  feature_std[feature_std < 1e-6] = 1.0
  target_mean, target_std = float(y_log.mean()), float(y_log.std())
  if target_std < 1e-6: target_std = 1.0
  x = (x - feature_mean) / feature_std
  y = (y_log - target_mean) / target_std
  model = CycleModel(feature_mean, feature_std, target_mean, target_std, masked_features=masked_features)
  optimizer = nn.optim.Adam(get_parameters(model.network), lr=config.learning_rate)
  rng, indexes, pairs = random.Random(config.seed), list(range(len(rows))), _pairs(rows)
  Tensor.manual_seed(config.seed)

  with Context(TRAINING=1):
    for _ in range(config.epochs):
      rng.shuffle(indexes)
      for start in range(0, len(indexes), config.batch_size):
        batch = indexes[start:start + config.batch_size]
        pred = model.network(Tensor(x[batch], device="CPU"))
        loss = _huber(pred - Tensor(y[batch], device="CPU"), config.huber_delta)
        if pairs:
          chosen = [pairs[rng.randrange(len(pairs))] for _ in range(min(len(batch), len(pairs)))]
          fast, slow = [p[0] for p in chosen], [p[1] for p in chosen]
          rank_loss = (-(model.network(Tensor(x[slow], device="CPU")) - model.network(Tensor(x[fast], device="CPU")))).softplus().mean()
          loss = loss + config.pairwise_weight * rank_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  predictions = model.network(Tensor(x, device="CPU")).numpy() * target_std + target_mean
  model.residual_rmse = float(np.sqrt(np.mean(np.square(predictions - y_log))))
  model.save(output, {"train_config":asdict(config), "training_records":len(rows)})
  return model
