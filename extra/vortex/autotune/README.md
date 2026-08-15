# TinyVortex neural HW/SW co-exploration

This research plugin learns a cycle model over a tinygrad UOp program, legal schedule, and parameterized Vortex configuration. It leaves tinygrad's optimizer and Tensor semantics unchanged. The initial hardware space is the 36-point cross product of 4/8/16/32 threads, 2/4/8 warps, and 1/2/4 cores, with one cluster and the default cache hierarchy.

The model optimizes SimX cycles only. A recommendation is not an area-, energy-, timing-, or cost-optimal hardware design.

## Development dependencies

The model itself only adds NumPy to tinygrad's normal runtime requirements. Validation uses the repository-local `venv`; the experiment environment was prepared with:

```bash
venv/bin/python -m pip install \
  'numpy>=2,<3' 'matplotlib>=3.9,<4' 'pytest>=8,<9' 'pytest-xdist>=3,<4' 'pytest-timeout>=2,<3' \
  'mypy==1.19.1' 'ruff==0.14.10'
```

The versions installed for the first complete experiment were NumPy 2.5.2, Matplotlib 3.11.1, pytest 8.4.2, pytest-xdist 3.8.0, pytest-timeout 2.4.0, mypy 1.19.1, and ruff 0.14.10. Matplotlib is used only to reproduce paper figures from the checked-in CSV artifacts.

## Workflow

Start Python without a nonnumeric `DEBUG` environment variable and configure the normal Vortex paths described in `tinyvortex.md`.

```bash
env -u DEBUG python3 -m extra.vortex.autotune collect \
  --output artifacts/vortex-autotune.jsonl

env -u DEBUG python3 -m extra.vortex.autotune train \
  --dataset artifacts/vortex-autotune.jsonl \
  --output artifacts/vortex-cycle-model.npz \
  --protocol joint

env -u DEBUG python3 -m extra.vortex.autotune evaluate \
  --dataset artifacts/vortex-autotune.jsonl \
  --model artifacts/vortex-cycle-model.npz \
  --protocol joint

env -u DEBUG python3 -m extra.vortex.autotune recommend \
  --model artifacts/vortex-cycle-model.npz --workload sgemm --shape 64,64,64

env -u DEBUG python3 -m extra.vortex.autotune run-model \
  --model artifacts/vortex-cycle-model.npz --tinymnist
```

Training and predictor inference are explicitly pinned to tinygrad's CPU backend, even when the collection shell uses `DEV=SIMX+VORTEX`. Vortex/SimX is used only to produce cycle labels and validate selected kernels. Add `--hardware-agnostic` to `train` to produce the paper's configuration-feature ablation model.

Full collection launches many isolated SimX processes. Use `--max-configs`, `--max-cases`, and `--candidate-limit` for a smoke dataset before starting the full experiment. Collection is append-only and resumes by workload, shape, hardware, schedule, and repeat number. Ten percent of design points are repeated by default to quantify simulator stability.

Focused experiments can repeat `--case`, and restrict comma-separated hardware axes. For example, the legacy CSV sweep is reproduced with `--case vecadd:512 --case vecadd:1024 --threads 4,8,16,32 --warps 4 --cores 1`.

The complete paper corpus (all breadth and SGEMM-depth cases, all 36 configurations, up to 24 schedules, and deterministic 10% repeats) is collected with:

```bash
env -u DEBUG DEV=SIMX+VORTEX \
  VORTEX_HOME=/home/nikhil/vortex \
  VORTEX_BUILD=/home/nikhil/vortex/build \
  LD_LIBRARY_PATH=/home/nikhil/vortex/build/sw/runtime \
  venv/bin/python -m extra.vortex.autotune collect \
  --output extra/vortex/autotune/results/full-simx.jsonl \
  --candidate-limit 24 --max-depth 6 --repeat-fraction 0.1 --workers 6 \
  --simx-timeout-ms 120000
```

Collection is batched by workload/configuration and the documented 12-core experiment uses six SimX workers for candidates within that fixed configuration. Hardware configurations remain sequential to prevent runtime rebuild races. Rerunning the same command resumes the JSONL rather than discarding completed measurements.
Each candidate is checkpointed immediately. The parent enforces a two-minute simulation budget plus a 30-second process/compilation allowance, so blocking, deadlocking, or pathological schedules are still given a feature-complete failed record without stalling the corpus.

### Two-hour CSV-comparable experiment

For the time-bounded first result, fix warps to 4 (matching the checked-in handwritten and TinyVortex CSVs), sweep threads and cores, retain six transformation-covering schedules, and hold out vecadd-4096 plus SGEMM-128 for inference:

```bash
env -u DEBUG DEV=SIMX+VORTEX \
  VORTEX_HOME=/home/nikhil/vortex \
  VORTEX_BUILD=/home/nikhil/vortex/build \
  LD_LIBRARY_PATH=/home/nikhil/vortex/build/sw/runtime \
  venv/bin/python -m extra.vortex.autotune collect \
  --output extra/vortex/autotune/results/two-hour-simx.jsonl \
  --case vecadd:512 --case vecadd:1024 --case vecadd:2048 --case vecadd:4096 \
  --case sgemm:32,32,32 --case sgemm:64,64,64 --case sgemm:128,128,128 \
  --threads 4,8,16,32 --warps 4 --cores 1,2,4 \
  --candidate-limit 6 --max-depth 3 --repeat-fraction 0.1 \
  --workers 6 --simx-timeout-ms 60000
```

Train on the five non-held-out shapes and evaluate the two held-out shapes with:

```bash
env -u DEBUG DEV=CPU venv/bin/python -m extra.vortex.autotune train \
  --dataset extra/vortex/autotune/results/two-hour-simx.jsonl \
  --output extra/vortex/autotune/results/two-hour-model.npz --protocol all \
  --exclude-case vecadd:4096 --exclude-case sgemm:128,128,128

env -u DEBUG DEV=CPU venv/bin/python -m extra.vortex.autotune evaluate \
  --dataset extra/vortex/autotune/results/two-hour-simx.jsonl \
  --model extra/vortex/autotune/results/two-hour-model.npz --protocol all \
  --only-case vecadd:4096 --only-case sgemm:128,128,128
```

The NPZ contains only MLP tensors. Its adjacent JSON file records the exact feature schema, normalization statistics, training settings, residual error, and dataset size. Recommendation confidence is a residual/novelty calibration score, not a probability of optimality.

For the final calibrated CPU model used by this pilot, reduce the ranking-loss weight so the output remains a useful cycle estimate as well as a ranking score:

```bash
env -u DEBUG DEV=CPU venv/bin/python -m extra.vortex.autotune train \
  --dataset extra/vortex/autotune/results/two-hour-simx.jsonl \
  --output extra/vortex/autotune/results/two-hour-model-calibrated.npz \
  --protocol all --epochs 300 --batch-size 128 \
  --learning-rate 0.0005 --pairwise-weight 0.05 --seed 42
```

The completed time-bounded corpus has 708 attempted measurements across seven shapes and 12 hardware configurations; 519 measurements were correct. On those measured designs, the calibrated model obtains 93.0% pairwise accuracy, 8.9% median absolute cycle error, 1.6% geometric-mean oracle regret, and 100% top-five oracle coverage. These are in-sample pilot metrics, not held-out claims. The shape-held-out model obtains 86.2% pairwise accuracy but does not reliably select hardware, which is the more honest generalization result from this small corpus.

The validated all-data recommendation for VecAdd-4096 selects 32 threads, four warps, one core, and `UPCAST(4) -> LOCAL(32)`. SimX measures 10,030 cycles, versus 21,184 cycles in `vecadd_tinyvortex.csv` and 40,595 cycles in `vecadd_vortex.csv` at 32 threads/four warps: 2.11x and 4.05x respectively. SGEMM-64 selects the current TinyVortex default at 487,926 cycles and does not beat the older CSV's 402,403-cycle TinyVortex point, so the pilot demonstrates a win on VecAdd rather than a universal win.

`run-model` executes the selected design in a fresh process. Before accepting a measurement, both collection and final execution query SimX capabilities and require threads, warps, cores, and clusters to match the design-point label. This prevents tinygrad's process-local device cache from silently reusing a previously opened hardware configuration.

### Clean VecAdd schedule experiment

The dedicated VecAdd schedule space enumerates every legal pair in the ordered form `UPCAST(U) -> LOCAL(L)`, with `U` in `{1,2,4,8,16}` and `L` a power of two up to `threads * warps`. A factor of one omits that transformation. Use `--schedule-space vecadd --candidate-limit 64` for collection and inference.

The clean experiment fixes one core and four warps, sweeps 4/8/16/32 threads, and collects sizes 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, and 8192. The benchmark sizes 512, 1024, 2048, and 4096 are excluded completely from CPU training. The resulting corpus contains 1,445 correct measurements and no failures.

On the four held-out benchmark sizes, the model obtains 95.1% pairwise accuracy, 7.0% median absolute cycle error, 3.9% geometric-mean oracle regret, and a 1.130x geometric-mean schedule-only speedup over current TinyVortex across all 16 size/thread points. Against the freshly measured handwritten Vortex regression, current TinyVortex reaches 1.481x and the held-out autotuner reaches 1.674x. See `vecadd_clean_heldout_comparison.csv` at the repository root for all cycles, schedules, and speedups.

### Clean SGEMM schedule experiment

The SGEMM-specific space models three distinct tiling decisions: output-column and output-row `UPCAST` factors (`UN` and `UM`), reduction-axis `UNROLL` (`UK`), and two-dimensional `LOCAL` factors (`LN` and `LM`). It also retains the unoptimized kernel, the current TinyVortex default, and legal `GROUP`/`GROUPTOP` variants. Output upcast products are capped at 16, reduction factors are drawn from `{1,2,4,8,16}`, and the local product is capped by `threads * warps`. Use `--schedule-space sgemm --candidate-limit 96` for this experiment.

The clean run fixes one core and four warps and sweeps 4/8/16/32 threads. Training uses only square sizes 24, 48, 80, and 96; benchmark sizes 32, 64, and 128 are excluded completely from CPU training. The saved corpus contains 2,802 attempted SimX measurements, of which 2,467 are numerically correct. The held-out model was trained from 1,489 valid training records for 200 epochs on `DEV=CPU` and is stored separately from the VecAdd model:

```bash
env -u DEBUG DEV=CPU venv/bin/python -m extra.vortex.autotune train \
  --dataset extra/vortex/autotune/results/sgemm-clean-simx.jsonl \
  --output extra/vortex/autotune/results/sgemm-clean-heldout-model.npz \
  --protocol all \
  --exclude-case sgemm:32,32,32 --exclude-case sgemm:64,64,64 \
  --exclude-case sgemm:112,112,112 --exclude-case sgemm:128,128,128 \
  --epochs 200 --batch-size 256 --learning-rate 0.0005 --pairwise-weight 0.05
```

Across the 12 fresh benchmark executions, the held-out policy is 2.966x faster than the updated TinyVortex default geometrically: six wins, four ties, and two losses. It is 13.992x faster at `N=32`, ties the default at `N=64`, and is 1.865x faster at `N=128`. The largest win is 23.37x at `N=128`, 32 threads. Against the freshly rebuilt handwritten Vortex SGEMM, its overall geometric mean is 0.684x, so this experiment establishes strong improvement over the generated baseline but not superiority to handwritten SGEMM. See `sgemm_clean_heldout_comparison.csv` at the repository root; schedule abbreviations use `UN/UM/UK/LN/LM` as defined above.
