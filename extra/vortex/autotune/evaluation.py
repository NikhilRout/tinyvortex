from __future__ import annotations

import hashlib, json, math, random
from collections import defaultdict
from typing import Iterable, Mapping

from extra.vortex.autotune.config import VortexHardwareConfig
from extra.vortex.autotune.model import CycleModel


def _shape_holdout(record: Mapping) -> bool:
  key = f"{record['workload']}:{','.join(map(str, record['shape']))}"
  return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 5 == 0


def _hardware_holdout(record: Mapping) -> bool:
  cfg = VortexHardwareConfig.from_dict(record["hardware"])
  return ((4, 8, 16, 32).index(cfg.threads) + (2, 4, 8).index(cfg.warps) + (1, 2, 4).index(cfg.cores)) % 4 == 0


def split_records(records: Iterable[Mapping], protocol: str = "joint") -> tuple[list[Mapping], list[Mapping]]:
  train, test = [], []
  for record in records:
    shape, hardware = _shape_holdout(record), _hardware_holdout(record)
    held = shape if protocol == "shape" else hardware if protocol == "hardware" else shape and hardware
    usable_train = not shape and not hardware if protocol == "joint" else not held
    if held: test.append(record)
    elif usable_train: train.append(record)
  return train, test


def _geomean(values: list[float]) -> float:
  return math.exp(sum(math.log(max(value, 1e-12)) for value in values) / len(values)) if values else float("nan")


def _hardware_key(record: Mapping) -> str: return VortexHardwareConfig.from_dict(record["hardware"]).key


def _aggregate_repeats(rows: list[Mapping], model: CycleModel) -> list[dict]:
  grouped: dict[tuple[str, str, str], list[Mapping]] = defaultdict(list)
  for row in rows:
    schedule = json.dumps(row["schedule"], sort_keys=True, separators=(",", ":"))
    grouped[(str(row["group_key"]), _hardware_key(row), schedule)].append(row)
  designs = []
  for (_, hardware, schedule), repeats in grouped.items():
    first = repeats[0]
    origins = {str(row.get("origin", "search")) for row in repeats}
    designs.append({"group":str(first["group_key"]), "hardware":hardware, "schedule":schedule,
                    "cycles":sum(float(row["cycles"]) for row in repeats) / len(repeats),
                    "predicted":model.predict(first["features"])[0], "default":"tinygrad-default" in origins,
                    "compile_seconds":sum(float(row.get("compile_seconds", 0.0)) for row in repeats) / len(repeats)})
  return designs


def evaluate_model(model: CycleModel, records: Iterable[Mapping]) -> dict[str, float | int]:
  """Evaluate schedule-only, hardware-only, and joint selection from measured design points."""
  rows = [x for x in records if x.get("correct", True) and float(x.get("cycles", 0)) > 0]
  designs = _aggregate_repeats(rows, model)
  groups: dict[str, list[dict]] = defaultdict(list)
  for design in designs: groups[design["group"]].append(design)

  schedule_speedups, joint_speedups, regrets, hardware_regrets, random_regrets = [], [], [], [], []
  pair_correct = pair_total = hardware_correct = top5_coverage = 0
  selected_hardware: set[str] = set()
  rng = random.Random(42)
  for values in groups.values():
    predicted_order = sorted(values, key=lambda item:item["predicted"])
    selected, oracle = predicted_order[0], min(values, key=lambda item:item["cycles"])
    defaults = [item for item in values if item["default"]]
    selected_hardware.add(selected["hardware"])
    regrets.append(selected["cycles"] / oracle["cycles"])
    hardware_correct += selected["hardware"] == oracle["hardware"]
    top5_coverage += any(item["hardware"] == oracle["hardware"] and item["schedule"] == oracle["schedule"] for item in predicted_order[:5])
    random_regrets.append(rng.choice(values)["cycles"] / oracle["cycles"])
    if defaults: joint_speedups.append(min(item["cycles"] for item in defaults) / selected["cycles"])

    by_hardware: dict[str, list[dict]] = defaultdict(list)
    for item in values: by_hardware[item["hardware"]].append(item)
    for hardware_values in by_hardware.values():
      chosen = min(hardware_values, key=lambda item:item["predicted"])
      default = next((item for item in hardware_values if item["default"]), None)
      if default is not None: schedule_speedups.append(default["cycles"] / chosen["cycles"])
    predicted_defaults = sorted(defaults, key=lambda item:item["predicted"])
    if predicted_defaults:
      hardware_regrets.append(predicted_defaults[0]["cycles"] / min(item["cycles"] for item in defaults))

    for i in range(len(values)):
      for j in range(i + 1, len(values)):
        actual = values[i]["cycles"] < values[j]["cycles"]
        predicted = values[i]["predicted"] < values[j]["predicted"]
        pair_correct += actual == predicted
        pair_total += 1

  repeat_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
  for row in rows:
    repeat_groups[(str(row["group_key"]), _hardware_key(row), json.dumps(row["schedule"], sort_keys=True))].append(float(row["cycles"]))
  stability = [(max(values) - min(values)) / (sum(values) / len(values)) for values in repeat_groups.values() if len(values) > 1]
  log_errors = [math.log(item["predicted"]) - math.log(item["cycles"]) for item in designs]
  absolute_percent_errors = sorted(abs(item["predicted"] / item["cycles"] - 1.0) for item in designs)
  median_ape = (absolute_percent_errors[len(absolute_percent_errors) // 2] if len(absolute_percent_errors) % 2 else
                sum(absolute_percent_errors[len(absolute_percent_errors) // 2 - 1:len(absolute_percent_errors) // 2 + 1]) / 2) \
               if absolute_percent_errors else float("nan")
  return {
    "records":len(rows), "distinct_designs":len(designs), "groups":len(groups),
    "schedule_only_geomean_speedup_vs_default":_geomean(schedule_speedups),
    "hardware_only_geomean_oracle_regret":_geomean(hardware_regrets),
    "joint_geomean_speedup_vs_best_default":_geomean(joint_speedups),
    # Backwards-compatible names used by early analysis notebooks.
    "geomean_speedup_vs_default":_geomean(joint_speedups), "geomean_oracle_regret":_geomean(regrets),
    "worst_case_regret":max(regrets, default=float("nan")), "random_joint_geomean_oracle_regret":_geomean(random_regrets),
    "hardware_recommendation_accuracy":hardware_correct / len(groups) if groups else float("nan"),
    "top5_oracle_coverage":top5_coverage / len(groups) if groups else float("nan"),
    "pairwise_accuracy":pair_correct / pair_total if pair_total else float("nan"),
    "log_cycle_rmse":math.sqrt(sum(x*x for x in log_errors) / len(log_errors)) if log_errors else float("nan"),
    "median_absolute_percent_cycle_error":median_ape,
    "selected_configuration_diversity":len(selected_hardware),
    "mean_compile_seconds":sum(item["compile_seconds"] for item in designs) / len(designs) if designs else float("nan"),
    "repeated_designs":len(stability), "mean_repeat_cycle_range":sum(stability) / len(stability) if stability else float("nan"),
  }
