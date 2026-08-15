from __future__ import annotations

import argparse, json, pathlib

import numpy as np

from extra.vortex.autotune.collect import JSON_MARKER, _worker, collect_dataset, describe_case, measure_batch, measure_case
from extra.vortex.autotune.config import VortexHardwareConfig, default_hardware_grid
from extra.vortex.autotune.data import read_jsonl
from extra.vortex.autotune.evaluation import evaluate_model, split_records
from extra.vortex.autotune.model import CycleModel, TrainConfig, train_cycle_model
from extra.vortex.autotune.policy import ApplicationRecommendation, KernelRecommendation, VortexAutoScheduler, rewrite_linear
from extra.vortex.autotune.runtime import execute_linear_measured, validate_runtime_config
from extra.vortex.autotune.search import deserialize_opts
from extra.vortex.autotune.workloads import all_cases, build_case, build_tinymnist, parse_case


def _case(args) -> object: return parse_case(args.workload, args.shape)


def _config(value: str) -> VortexHardwareConfig: return VortexHardwareConfig.from_dict(json.loads(value))


def _selected_grid(args) -> tuple[VortexHardwareConfig, ...]:
  configs = default_hardware_grid()
  for name in ("threads", "warps", "cores"):
    if value := getattr(args, name, None):
      allowed = set(map(int, value.split(",")))
      configs = tuple(config for config in configs if getattr(config, name) in allowed)
  return configs[:args.max_configs] if getattr(args, "max_configs", None) else configs


def _add_case(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--workload", required=True, choices=("vecadd", "reduction", "sgemm", "conv"))
  parser.add_argument("--shape", required=True, help="comma-separated workload dimensions")


def _add_hardware_filter(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--threads", help="comma-separated thread counts from 4,8,16,32")
  parser.add_argument("--warps", help="comma-separated warp counts from 2,4,8")
  parser.add_argument("--cores", help="comma-separated core counts from 1,2,4")
  parser.add_argument("--max-configs", type=int)


def _case_spec(value: str):
  try: name, shape = value.split(":", 1)
  except ValueError as exc: raise argparse.ArgumentTypeError("case must be NAME:DIM[,DIM...]") from exc
  try: return parse_case(name, shape)
  except (ValueError, KeyError) as exc: raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="TinyVortex neural hardware/software co-exploration")
  commands = parser.add_subparsers(dest="command", required=True)
  collect = commands.add_parser("collect", help="collect a resumable SimX JSONL dataset in isolated workers")
  collect.add_argument("--output", required=True)
  collect.add_argument("--workloads", default="vecadd,reduction,sgemm,conv")
  collect.add_argument("--case", action="append", type=_case_spec, dest="cases",
                       help="explicit NAME:DIM[,DIM...] case; repeat to replace the built-in suite")
  collect.add_argument("--candidate-limit", type=int, default=24)
  collect.add_argument("--max-depth", type=int, default=6)
  collect.add_argument("--schedule-space", choices=("generic", "vecadd", "sgemm"), default="generic")
  _add_hardware_filter(collect)
  collect.add_argument("--max-cases", type=int)
  collect.add_argument("--no-sgemm-depth", action="store_true")
  collect.add_argument("--repeat-fraction", type=float, default=0.1)
  collect.add_argument("--simx-timeout-ms", type=int, default=120_000,
                       help="per-kernel simulator timeout; failed candidates remain recorded")
  collect.add_argument("--workers", type=int, default=4, help="parallel SimX workers within one fixed hardware configuration")

  train = commands.add_parser("train", help="train the structured tinygrad MLP")
  train.add_argument("--dataset", required=True)
  train.add_argument("--output", required=True)
  train.add_argument("--protocol", choices=("shape", "hardware", "joint", "all"), default="joint")
  train.add_argument("--epochs", type=int, default=100)
  train.add_argument("--batch-size", type=int, default=256)
  train.add_argument("--learning-rate", type=float, default=1e-3)
  train.add_argument("--pairwise-weight", type=float, default=0.25)
  train.add_argument("--seed", type=int, default=42)
  train.add_argument("--exclude-case", action="append", type=_case_spec, default=[],
                     help="exclude a held-out NAME:DIM[,DIM...] case from training")
  train.add_argument("--hardware-agnostic", action="store_true",
    help="train the ablation baseline with all Vortex configuration features masked")

  evaluate = commands.add_parser("evaluate", help="evaluate ranking and oracle regret")
  evaluate.add_argument("--dataset", required=True)
  evaluate.add_argument("--model", required=True)
  evaluate.add_argument("--protocol", choices=("shape", "hardware", "joint", "all"), default="joint")
  evaluate.add_argument("--only-case", action="append", type=_case_spec, default=[],
                        help="evaluate only a NAME:DIM[,DIM...] held-out case")

  recommend = commands.add_parser("recommend", help="recommend a hardware configuration and schedule")
  _add_case(recommend)
  recommend.add_argument("--model", required=True)
  recommend.add_argument("--top-k", type=int, default=5)
  recommend.add_argument("--candidate-limit", type=int, default=24)
  recommend.add_argument("--schedule-space", choices=("generic", "vecadd", "sgemm"), default="generic")
  _add_hardware_filter(recommend)

  run = commands.add_parser("run-model", help="recommend, execute, and validate a microbenchmark or TinyMNIST")
  run.add_argument("--model", required=True)
  run.add_argument("--workload", choices=("vecadd", "reduction", "sgemm", "conv"))
  run.add_argument("--shape")
  run.add_argument("--tinymnist", action="store_true")
  run.add_argument("--top-k", type=int, default=5)
  run.add_argument("--candidate-limit", type=int, default=24)
  run.add_argument("--schedule-space", choices=("generic", "vecadd", "sgemm"), default="generic")
  _add_hardware_filter(run)

  execute = commands.add_parser("worker-run-recommendation", help=argparse.SUPPRESS)
  execute.add_argument("--recommendation", required=True)
  execute.add_argument("--workload", choices=("vecadd", "reduction", "sgemm", "conv"))
  execute.add_argument("--shape")
  execute.add_argument("--tinymnist", action="store_true")

  describe = commands.add_parser("worker-describe", help=argparse.SUPPRESS)
  _add_case(describe)
  describe.add_argument("--config", required=True)
  describe.add_argument("--candidate-limit", type=int, default=24)
  describe.add_argument("--max-depth", type=int, default=6)
  describe.add_argument("--schedule-space", choices=("generic", "vecadd", "sgemm"), default="generic")
  measure = commands.add_parser("worker-measure", help=argparse.SUPPRESS)
  _add_case(measure)
  measure.add_argument("--config", required=True)
  measure.add_argument("--call-index", type=int, required=True)
  measure.add_argument("--schedule", required=True)
  measure.add_argument("--origin", default="search")
  measure.add_argument("--repeat", type=int, default=0)
  measure_batch_parser = commands.add_parser("worker-measure-batch", help=argparse.SUPPRESS)
  _add_case(measure_batch_parser)
  measure_batch_parser.add_argument("--config", required=True)
  measure_batch_parser.add_argument("--requests", required=True)
  measure_batch_parser.add_argument("--checkpoint")
  measure_batch_parser.add_argument("--timeout-ms", type=int, default=120_000)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  if args.command == "worker-describe":
    print(JSON_MARKER + json.dumps(describe_case(_case(args), _config(args.config), args.candidate_limit, args.max_depth,
                                                args.schedule_space)))
    return 0
  if args.command == "worker-measure":
    record = measure_case(_case(args), _config(args.config), args.call_index, json.loads(args.schedule), args.origin, args.repeat)
    print(JSON_MARKER + json.dumps(record.to_dict(), sort_keys=True))
    return 0
  if args.command == "worker-measure-batch":
    import os
    os.environ["VORTEX_TIMEOUT"] = str(args.timeout_ms)
    records = measure_batch(_case(args), _config(args.config), json.loads(args.requests), args.checkpoint)
    print(JSON_MARKER + json.dumps({"records":len(records), "correct":all(record["correct"] for record in records)}, sort_keys=True))
    return 0
  if args.command == "worker-run-recommendation":
    data, config = json.loads(args.recommendation), None
    config = VortexHardwareConfig.from_dict(data["hardware"])
    kernels = tuple(KernelRecommendation(int(row["call_index"]), str(row["ast_key"]), deserialize_opts(row["schedule"]),
                                         float(row["predicted_cycles"]), float(row["confidence"])) for row in data["kernels"])
    recommendation = ApplicationRecommendation(config, kernels, float(data["predicted_cycles"]), float(data["confidence"]))
    with config.environment(build_runtime=True):
      if args.tinymnist: output, reference = build_tinymnist()
      else:
        if args.workload is None or args.shape is None: raise ValueError("worker requires a workload and shape")
        output, reference = build_case(parse_case(args.workload, args.shape), realize_inputs=False)
      linear, var_vals = output.linear_with_vars()
      caps = validate_runtime_config(config)
      measured = execute_linear_measured(rewrite_linear(linear, recommendation), var_vals)
      np.testing.assert_allclose(output.numpy(), reference.numpy(), rtol=1e-3, atol=1e-3)
    print(JSON_MARKER + json.dumps({"measured_cycles":measured.total_cycles, "measured_instructions":measured.total_instructions,
                                   "runtime_caps":caps, "correct":True}, sort_keys=True))
    return 0
  if args.command == "collect":
    allowed = set(args.workloads.split(","))
    cases = list(args.cases) if args.cases else [case for case in all_cases(not args.no_sgemm_depth) if case.name in allowed]
    if args.max_cases: cases = cases[:args.max_cases]
    import os
    os.environ["VORTEX_TIMEOUT"] = str(args.simx_timeout_ms)
    collect_dataset(cases, _selected_grid(args), args.output, args.candidate_limit, args.max_depth,
                    args.repeat_fraction, args.simx_timeout_ms, args.workers, args.schedule_space)
    return 0
  if args.command == "train":
    rows = list(read_jsonl(args.dataset))
    excluded = {(case.name, case.shape) for case in args.exclude_case}
    if excluded: rows = [row for row in rows if (row["workload"], tuple(row["shape"])) not in excluded]
    training = rows if args.protocol == "all" else split_records(rows, args.protocol)[0]
    cfg = TrainConfig(args.epochs, args.batch_size, args.learning_rate, args.pairwise_weight, seed=args.seed,
                      hardware_agnostic=args.hardware_agnostic)
    train_cycle_model(training, args.output, cfg)
    print(json.dumps({"training_records":len(training), "model":str(pathlib.Path(args.output).resolve())}, indent=2))
    return 0
  if args.command == "evaluate":
    rows = list(read_jsonl(args.dataset))
    included = {(case.name, case.shape) for case in args.only_case}
    if included: rows = [row for row in rows if (row["workload"], tuple(row["shape"])) in included]
    testing = rows if args.protocol == "all" else split_records(rows, args.protocol)[1]
    print(json.dumps(evaluate_model(CycleModel.load(args.model), testing), indent=2, sort_keys=True))
    return 0
  if args.command == "recommend":
    output, _ = build_case(_case(args), realize_inputs=False)
    linear, var_vals = output.linear_with_vars()
    policy = VortexAutoScheduler(args.model, schedule_limit=args.candidate_limit, schedule_space=args.schedule_space)
    print(json.dumps(policy.recommend_application(linear, _selected_grid(args), args.top_k, var_vals).to_dict(), indent=2))
    return 0
  if args.command == "run-model":
    if args.tinymnist:
      output, reference = build_tinymnist()
    else:
      if args.workload is None or args.shape is None: raise ValueError("run-model requires --tinymnist or both --workload and --shape")
      output, reference = build_case(_case(args), realize_inputs=False)
    linear, var_vals = output.linear_with_vars()
    policy = VortexAutoScheduler(args.model, schedule_limit=args.candidate_limit, schedule_space=args.schedule_space)
    recommendation = policy.recommend_application(linear, _selected_grid(args), args.top_k, var_vals)
    command = ["worker-run-recommendation", "--recommendation", json.dumps(recommendation.to_dict())]
    if args.tinymnist: command.append("--tinymnist")
    else: command += ["--workload", args.workload, "--shape", args.shape]
    execution = _worker(command)
    result = {**recommendation.to_dict(), **execution}
    print(json.dumps(result, indent=2))
    return 0
  raise AssertionError(args.command)
