import json, tempfile, unittest
from pathlib import Path

from tinygrad import Tensor
from tinygrad.helpers import Context, Target
from tinygrad.nn.state import get_parameters
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops

from extra.vortex.autotune.config import VortexHardwareConfig, default_hardware_grid
from extra.vortex.autotune.data import MeasurementRecord, append_jsonl, read_jsonl
from extra.vortex.autotune.evaluation import evaluate_model
from extra.vortex.autotune.features import FEATURE_NAMES, extract_features
from extra.vortex.autotune.model import CycleModel
from extra.vortex.autotune.policy import ApplicationRecommendation, KernelRecommendation, VortexAutoScheduler, rewrite_linear
from extra.vortex.autotune.search import (deserialize_opts, enumerate_schedules, enumerate_sgemm_schedules, enumerate_vecadd_schedules,
                                          replay_schedule, serialize_opts)


def make_linear():
  source = Tensor.rand(8, 8, device="CPU").realize()
  output = (source + source).sum(axis=1)
  linear = output.schedule_linear()
  ast = next(call.src[0] for call in linear.src if call.op is Ops.CALL and call.src[0].op is Ops.SINK)
  return linear, ast


class FakeCycleModel:
  def predict(self, features): return 100.0 + sum(features[-8:]), 0.9


class TestVortexAutotune(unittest.TestCase):
  def setUp(self):
    self.config = VortexHardwareConfig(4, 2, 1)
    self.renderer = Renderer(Target(device="FAKE"))

  def test_hardware_grid(self):
    grid = default_hardware_grid()
    self.assertEqual(len(grid), 36)
    self.assertEqual(len(set(grid)), 36)
    self.assertEqual(self.config.resident_threads, 8)
    with self.assertRaises(ValueError): VortexHardwareConfig(3, 2, 1)

  def test_schedule_enumeration_and_features(self):
    _, ast = make_linear()
    candidates = enumerate_schedules(ast, self.renderer, limit=8, max_depth=2, seed=1)
    self.assertGreaterEqual(len(candidates), 2)
    self.assertEqual(candidates[0].origin, "unoptimized")
    self.assertEqual(len({x.opts for x in candidates}), len(candidates))
    schedule = candidates[-1].opts
    self.assertEqual(deserialize_opts(serialize_opts(schedule)), schedule)
    values = extract_features(ast, replay_schedule(ast, self.renderer, schedule), schedule, self.config)
    self.assertEqual(len(values), len(FEATURE_NAMES))
    self.assertTrue(all(isinstance(x, float) for x in values))

  def test_vecadd_schedule_product(self):
    source = Tensor.rand(512, device="CPU").realize()
    linear = (source + source).schedule_linear()
    ast = next(call.src[0] for call in linear.src if call.op is Ops.CALL and call.src[0].op is Ops.SINK)
    renderer = Renderer(Target(device="FAKE"))
    renderer.local_max = (32, 1, 1)
    schedules = enumerate_vecadd_schedules(ast, renderer)
    serialized = [serialize_opts(x.opts) for x in schedules]
    self.assertIn([{"op":"UPCAST", "axis":0, "arg":8}, {"op":"LOCAL", "axis":0, "arg":32}], serialized)
    self.assertEqual(len(schedules), 30)

  def test_sgemm_structured_sample(self):
    a, b = Tensor.rand(32, 32, device="CPU").realize(), Tensor.rand(32, 32, device="CPU").realize()
    linear = (a @ b).schedule_linear()
    ast = next(call.src[0] for call in linear.src if call.op is Ops.CALL and call.src[0].op is Ops.SINK)
    renderer = Renderer(Target(device="FAKE"))
    renderer.local_max = (32, 1, 1)
    schedules = enumerate_sgemm_schedules(ast, renderer, limit=160)
    self.assertEqual(len(schedules), 160)
    self.assertTrue(any(any(opt.op.name == "UNROLL" and opt.arg == 8 for opt in candidate.opts) for candidate in schedules))
    self.assertTrue(any(sum(opt.op.name == "LOCAL" for opt in candidate.opts) == 2 for candidate in schedules))

  def test_policy_and_rewrite(self):
    linear, ast = make_linear()
    policy = VortexAutoScheduler(FakeCycleModel(), renderer_factory=lambda _:self.renderer, schedule_limit=6, max_depth=1)
    selected = policy.select_schedule(ast, self.config)
    self.assertGreater(selected.predicted_cycles, 0)
    call_index = next(i for i,c in enumerate(linear.src) if c.op is Ops.CALL and c.src[0] is ast)
    kernel = KernelRecommendation(call_index, ast.key.hex(), selected.schedule, selected.predicted_cycles, selected.confidence)
    recommendation = ApplicationRecommendation(self.config, (kernel,), selected.predicted_cycles, selected.confidence)
    rewritten = rewrite_linear(linear, recommendation)
    self.assertEqual(rewritten.src[call_index].src[0].arg.opts_to_apply, selected.schedule)

  def test_model_artifact_roundtrip(self):
    # Collection is normally launched with a Vortex DEV; predictor creation
    # must remain pinned to CPU regardless of that ambient setting.
    with Context(DEV="NPY"): model = CycleModel()
    self.assertEqual({parameter.device for parameter in get_parameters(model.network)}, {"CPU"})
    features = [0.0] * len(FEATURE_NAMES)
    expected = model.predict(features)[0]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model.npz"
      model.save(path, {"purpose":"test"})
      loaded = CycleModel.load(path)
      self.assertAlmostEqual(loaded.predict(features)[0], expected, places=5)
      self.assertEqual(json.loads(path.with_suffix(".json").read_text())["purpose"], "test")

  def test_jsonl_and_evaluation(self):
    features = [0.0] * len(FEATURE_NAMES)
    records = [MeasurementRecord("vecadd", (256,), self.config, "ast", [], features, cycles, cycles // 2, True,
                                 "tinygrad-default" if cycles == 120 else "search") for cycles in (120, 100)]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "measurements.jsonl"
      append_jsonl(path, records)
      loaded = list(read_jsonl(path))
      self.assertEqual(len(loaded), 2)
      metrics = evaluate_model(FakeCycleModel(), loaded)
      self.assertEqual(metrics["groups"], 1)
      self.assertGreater(metrics["geomean_oracle_regret"], 0)


if __name__ == "__main__": unittest.main()
