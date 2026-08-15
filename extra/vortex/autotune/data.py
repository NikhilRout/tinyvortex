from __future__ import annotations

import json, os, pathlib, subprocess
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Mapping

from extra.vortex.autotune.config import VortexHardwareConfig

SCHEMA_VERSION = 1


def git_revision(path: str | pathlib.Path) -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, stderr=subprocess.DEVNULL, text=True).strip()
  except (OSError, subprocess.CalledProcessError): return "unknown"


@dataclass(frozen=True)
class MeasurementRecord:
  workload: str
  shape: tuple[int, ...]
  hardware: VortexHardwareConfig
  ast_key: str
  schedule: list[dict]
  features: list[float]
  cycles: int
  instructions: int
  correct: bool
  origin: str = "search"
  source_hash: str = ""
  compile_seconds: float = 0.0
  tinyvortex_revision: str = "unknown"
  vortex_revision: str = "unknown"
  repeat: int = 0
  error: str = ""

  @property
  def group_key(self) -> str: return f"{self.workload}:{','.join(map(str, self.shape))}:{self.ast_key}"

  def to_dict(self) -> dict:
    return {"schema_version":SCHEMA_VERSION, **asdict(self), "hardware":self.hardware.to_dict(), "group_key":self.group_key}

  @staticmethod
  def from_dict(value: Mapping) -> MeasurementRecord:
    data = dict(value)
    if int(data.pop("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION: raise ValueError("unsupported measurement schema")
    data.pop("group_key", None)
    data["shape"], data["hardware"] = tuple(data["shape"]), VortexHardwareConfig.from_dict(data["hardware"])
    return MeasurementRecord(**data)


def read_jsonl(path: str | pathlib.Path) -> Iterator[dict]:
  source = pathlib.Path(path)
  if not source.exists(): return
  with source.open() as stream:
    for number, line in enumerate(stream, 1):
      if not line.strip(): continue
      try: yield json.loads(line)
      except json.JSONDecodeError as exc: raise ValueError(f"invalid JSON at {source}:{number}") from exc


def append_jsonl(path: str | pathlib.Path, records: Iterable[MeasurementRecord | Mapping]) -> None:
  output = pathlib.Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("a") as stream:
    for record in records:
      value = record.to_dict() if isinstance(record, MeasurementRecord) else dict(record)
      stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def completed_keys(path: str | pathlib.Path) -> set[tuple]:
  return {(x["workload"], tuple(x["shape"]), VortexHardwareConfig.from_dict(x["hardware"]).key,
           json.dumps(x["schedule"], sort_keys=True), int(x.get("repeat", 0))) for x in read_jsonl(path)}
