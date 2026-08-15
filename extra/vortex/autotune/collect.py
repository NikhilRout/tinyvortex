from __future__ import annotations

import fcntl, hashlib, json, os, pathlib, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

import numpy as np

from tinygrad.device import Device
from tinygrad.uop.ops import Ops

from extra.vortex.autotune.config import VortexHardwareConfig
from extra.vortex.autotune.data import MeasurementRecord, append_jsonl, completed_keys, git_revision
from extra.vortex.autotune.features import extract_features
from extra.vortex.autotune.policy import KernelRecommendation, rewrite_linear
from extra.vortex.autotune.runtime import execute_linear_measured, validate_runtime_config
from extra.vortex.autotune.search import deserialize_opts, enumerate_schedules, replay_schedule, schedule_local_size, serialize_opts
from extra.vortex.autotune.workloads import WorkloadCase, build_case

JSON_MARKER = "AUTOTUNE_JSON="
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNTIME_LOCK = pathlib.Path(tempfile.gettempdir()) / "tinyvortex-autotune-runtime.lock"


def _kernel_calls(linear):
  return [(i, call.src[0]) for i, call in enumerate(linear.src) if call.op is Ops.CALL and call.device == "VORTEX" and
          call.src and call.src[0].op is Ops.SINK]


def describe_case(case: WorkloadCase, config: VortexHardwareConfig, limit: int = 24, max_depth: int = 6,
                  schedule_space: str = "generic") -> list[dict]:
  with config.environment(build_runtime=True):
    output, _ = build_case(case)
    linear, _ = output.linear_with_vars()
    renderer = Device["VORTEX"].renderer
    return [{"call_index":index, "ast_key":ast.key.hex(),
             "candidates":[{"origin":candidate.origin, "schedule":serialize_opts(candidate.opts)}
                           for candidate in enumerate_schedules(ast, renderer, limit, max_depth, schedule_space=schedule_space)]}
            for index, ast in _kernel_calls(linear)]


def measure_case(case: WorkloadCase, config: VortexHardwareConfig, call_index: int, schedule_data: Sequence[dict],
                 origin: str = "search", repeat: int = 0, forced_error: str = "") -> MeasurementRecord:
  opts = deserialize_opts(schedule_data)
  with config.environment(build_runtime=True):
    # All candidates for a configuration share Vortex's build directory. The
    # make step must be serialized, but SimX execution is parallel after open.
    with RUNTIME_LOCK.open("w") as lock:
      fcntl.flock(lock, fcntl.LOCK_EX)
      output, reference = build_case(case)
      validate_runtime_config(config)
      fcntl.flock(lock, fcntl.LOCK_UN)
    linear, var_vals = output.linear_with_vars()
    calls = dict(_kernel_calls(linear))
    if call_index not in calls: raise ValueError(f"case has no kernel call {call_index}")
    ast, renderer = calls[call_index], Device["VORTEX"].renderer
    scheduler = replay_schedule(ast, renderer, opts)
    features = extract_features(ast, scheduler, opts, config, var_vals)
    recommendation = KernelRecommendation(call_index, ast.key.hex(), opts, 0.0, 0.0)
    rewritten = rewrite_linear(linear, (recommendation,))
    error, correct = "", True
    try:
      if forced_error: raise RuntimeError(forced_error)
      local_size = schedule_local_size(scheduler)
      if local_size > config.local_size_limit:
        raise RuntimeError(f"local size {local_size} exceeds hardware limit {config.local_size_limit}")
      measured = execute_linear_measured(rewritten, var_vals, (call_index,))
      np.testing.assert_allclose(output.numpy(), reference.numpy(), rtol=1e-3, atol=1e-3)
    except Exception as exc:
      measured, correct, error = None, False, f"{type(exc).__name__}: {exc}"
    counters = measured.per_call.get(call_index) if measured is not None else None
    vortex_home = pathlib.Path(os.getenv("VORTEX_HOME", pathlib.Path.home() / "vortex"))
    return MeasurementRecord(case.name, case.shape, config, ast.key.hex(), list(schedule_data), features,
                             counters.cycles if counters else 0, counters.instructions if counters else 0, correct, origin,
                             measured.source_hashes.get(call_index, "") if measured else "", measured.compile_seconds if measured else 0.0,
                             git_revision(REPO_ROOT), git_revision(vortex_home), repeat, error)


def measure_batch(case: WorkloadCase, config: VortexHardwareConfig, requests: Sequence[dict],
                  checkpoint: str | pathlib.Path | None = None) -> list[dict]:
  """Measure one case/configuration batch while reusing a single SimX runtime process."""
  records = []
  for index, request in enumerate(requests, 1):
    record = measure_case(case, config, int(request["call_index"]), request["schedule"],
                          str(request.get("origin", "search")), int(request.get("repeat", 0)),
                          str(request.get("forced_error", "")))
    value = record.to_dict()
    records.append(value)
    if checkpoint is not None: append_jsonl(checkpoint, (value,))
    print(f"  candidate {index}/{len(requests)}: {'ok' if record.correct else record.error}", flush=True)
    if "vx_event_wait_value" in record.error: break
  return records


class WorkerTimeout(RuntimeError): pass


def _worker(command: Sequence[str], timeout_seconds: float | None = None) -> object:
  env = os.environ.copy()
  env.pop("DEBUG", None)
  env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  try:
    result = subprocess.run([sys.executable, "-m", "extra.vortex.autotune", *command], cwd=REPO_ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout_seconds)
  except subprocess.TimeoutExpired as exc:
    raise WorkerTimeout(f"worker exceeded {timeout_seconds:.0f}s wall-clock timeout") from exc
  marked = [line[len(JSON_MARKER):] for line in result.stdout.splitlines() if line.startswith(JSON_MARKER)]
  if result.returncode or not marked: raise RuntimeError(f"autotune worker failed ({result.returncode})\n{result.stdout}")
  return json.loads(marked[-1])


def collect_dataset(cases: Iterable[WorkloadCase], configs: Iterable[VortexHardwareConfig], output: str | pathlib.Path,
                    limit: int = 24, max_depth: int = 6, repeat_fraction: float = 0.1, timeout_ms: int = 120_000,
                    workers: int = 4, schedule_space: str = "generic") -> None:
  if not 0.0 <= repeat_fraction <= 1.0: raise ValueError("repeat_fraction must be between zero and one")
  if workers < 1: raise ValueError("workers must be positive")
  done, cases, configs = completed_keys(output), tuple(cases), tuple(configs)
  total, current = len(cases) * len(configs), 0
  for config in configs:
    for case in cases:
      current += 1
      description = _worker(("worker-describe", "--workload", case.name, "--shape", ",".join(map(str, case.shape)),
                             "--config", json.dumps(config.to_dict()), "--candidate-limit", str(limit), "--max-depth", str(max_depth),
                             "--schedule-space", schedule_space))
      requests = []
      for kernel in description:
        for candidate in kernel["candidates"]:
          schedule_json = json.dumps(candidate["schedule"], sort_keys=True)
          repeat_hash = int(hashlib.sha256(f"{case.key}:{config.key}:{schedule_json}".encode()).hexdigest(), 16) % 10_000
          repeats = 2 if repeat_hash < repeat_fraction * 10_000 else 1
          for repeat in range(repeats):
            key = (case.name, case.shape, config.key, schedule_json, repeat)
            if key in done: continue
            requests.append({"call_index":kernel["call_index"], "schedule":candidate["schedule"],
                             "origin":candidate["origin"], "repeat":repeat, "key":key})
      print(f"[{current}/{total}] {case.key} {config.key}: {len(requests)} pending measurements", flush=True)
      if not requests: continue
      case_timeout = timeout_ms

      def run_request(request: dict) -> tuple[tuple, bool]:
        command = ("worker-measure-batch", "--workload", case.name, "--shape", ",".join(map(str, case.shape)),
                   "--config", json.dumps(config.to_dict()), "--requests", json.dumps([request]),
                   "--checkpoint", str(pathlib.Path(output).resolve()), "--timeout-ms", str(case_timeout))
        try: result = _worker(command, timeout_seconds=case_timeout / 1000 + 30)
        except WorkerTimeout as exc:
          failed = {**request, "forced_error":str(exc)}
          result = _worker((*command[:command.index("--requests") + 1], json.dumps([failed]), *command[command.index("--requests") + 2:]))
        if int(result["records"]) != 1: raise RuntimeError("measurement worker did not record exactly one candidate")
        return request["key"], bool(result["correct"])

      schedule_groups: dict[str, list[dict]] = {}
      for request in requests:
        schedule_groups.setdefault(json.dumps(request["schedule"], sort_keys=True), []).append(request)
      chunks = [[] for _ in range(min(workers, len(schedule_groups)))]
      for index, group in enumerate(schedule_groups.values()): chunks[index % len(chunks)].extend(group)

      def run_chunk(chunk: list[dict]) -> list[tuple]:
        finished, base_correct = [], True
        for request in chunk:
          current = request if int(request.get("repeat", 0)) == 0 or base_correct else {
            **request, "forced_error":"stability repeat skipped because the base measurement failed"}
          key, correct = run_request(current)
          finished.append(key)
          if int(request.get("repeat", 0)) == 0: base_correct = correct
        return finished

      with ThreadPoolExecutor(max_workers=workers) as pool:
        for finished in pool.map(run_chunk, chunks): done.update(finished)
