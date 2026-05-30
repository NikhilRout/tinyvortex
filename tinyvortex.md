# Tinyvortex

Tinyvortex is a [tinygrad](https://github.com/tinygrad/tinygrad) fork with a native `VORTEX` backend to easily compile and run Tensor programs on the open-source RISC-V based [Vortex GPGPU](https://github.com/vortexgpgpu/vortex) architecture. This platform serves as an experimental framework for running, profiling and analyzing MLSys workloads on Vortex, and guides future GPU microarchitecture design-space exploration. 

tinygrad programs still have the standard Tensor API, while generated kernels are rendered as Vortex C++ intrinsics, compiled to `.vxbin`, and launched through the Vortex C runtime with Python `ctypes`.

The default target is `simx` (cycle-approx simulator) to keep simulation times reasonable during development and testing. However, `rtlsim` (verilator based functional RTL simulation) and WIP `XRT/xrtsim` (AMD Xilinx FPGA) drivers are also supported for accurate hardware benchmarking. 

## Initial Setup

1. Setup a `/build` directory in the Vortex repo as per project [guidelines](https://github.com/vortexgpgpu/vortex/blob/master/docs/install_vortex.md)

2. Clone tinyvortex and enter the directory:
```
git clone https://github.com/NikhilRout/tinyvortex.git
cd tinyvortex
```
3. Create and activate a Python virtual environment, and install tinyvortex:
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .
```
4. Point tinyvortex to Vortex runtime:
```bash
export VORTEX_HOME=$HOME/vortex
export VORTEX_BUILD=$VORTEX_HOME/build
export LD_LIBRARY_PATH=$VORTEX_BUILD/sw/runtime:$LD_LIBRARY_PATH
```

5. Run an example smoke test to verify setup:
```bash
DEV=VORTEX python3 - <<'PY'
from tinygrad import Tensor
print((Tensor([1, 2, 3], device="VORTEX") * 2).tolist())
PY
```

## Hardware Configs and Running Tensor Programs

Tinyvortex supports all Vortex runtime drivers; choose one through `DEV`:
```bash
DEV=VORTEX         # default: simx
DEV=SIMX+VORTEX    # cycle-approx simulator
DEV=RTLSIM+VORTEX  # verilator-based RTL simulator
DEV=XRTSIM+VORTEX  # Xilinx FPGA, WIP
```

Vortex hardware configs are passed through `VORTEX_CONFIGS` and forwarded to the Vortex kernel build. Example:
```bash
DEV=VORTEX VORTEX_CONFIGS='-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_NUM_WARPS=4 -DVX_CFG_NUM_THREADS=32' VORTEX_BUILD_RUNTIME=1 python3 examples/vortex/vecadd.py
``` 
- Use `VORTEX_BUILD_RUNTIME=1` when changing runtime/simulator-affecting configs
- Use `VORTEX_REBUILD=1` to clean the tinyvortex kernel build directory before compiling.
- `VORTEX_CONFIGS` is included in the tinygrad compile cache key and forwarded to the Vortex kernel build
- When running `DEV=RTLSIM+VORTEX` with Verilator builds that treat RTL warnings as fatal, add `-Wno-fatal` to `VORTEX_CONFIGS`
- Add `BROWSER=1 VIZ=1` to launch tinygrad's UOp graph/rewrite visualizer after the run
- Use `JITBEAM=4` with `@TinyJit` to spend more capture time searching faster kernels for repeated runs

Try out lightweight MNIST inference on Vortex!
```bash
DEV=VORTEX VORTEX_CONFIGS='-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_NUM_WARPS=4 -DVX_CFG_NUM_THREADS=32' python3 examples/vortex/tinymnist.py --infer-samples=4
```

Other useful knobs:
```bash
VORTEX_KERNEL_LIB=vortex2
VORTEX_MAKE_ARGS='-j8'
VORTEX_TIMEOUT=600000
VORTEX_DEBUG=1        # forwarded to Vortex make as DEBUG
VORTEX_PERF=1         # mirrors Vortex blackbox --perf=1
VORTEX_SCOPE=1
VORTEX_SAIF=1
```

## Utilizing Vortex Tensor Cores

- Add -DEXT_TCU_ENABLE to `VORTEX_CONFIGS` to include the Vortex Tensor Core Unit extension in your build
- Vortex TCUs are mixed-precision; low-precision input formats are selected by argpassing `--itype` and the higher-precision accumulator/output format is selected through `--otype`
- Matrix dimensions can be specified with `--m`, `--n`, and `--base-k` (fp32 dim = 2x fp16 = 4x fp8 = 8x fp4)
- Example:
```bash
VORTEX_CONFIGS='-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_NUM_THREADS=32' VORTEX_BUILD_RUNTIME=1 python3 examples/vortex/sgemm_tcu.py --itype=bf16 --otype=fp32 --m=64 --n=64 --base-k=64
```
- Use `TC=0` to force scalar/SIMT lowering if you want to compare against non-TCU codegen

Note: 4-bit datatype handling, structured sparsity, microscaling formats, WGMMA, DXA, and async barriers are WIP to be ported soon.

## Tinyvortex's Tinygrad Extension Code Map

- `tinygrad/device.py`: registers `VORTEX` as a device.
- `tinygrad/runtime/ops_vortex.py`: backend implementation.
  - `VortexConfig` resolves source-tree vs SDK mode, driver, config strings, paths, and cache keys.
  - `VortexRenderer` renders UOps to Vortex C++ and owns VORTEX-specific tensor-core descriptors/lowering.
  - `VortexCompiler` invokes `extra/vortex/Makefile.kernel`.
  - `VortexRuntime`, `VortexAllocator`, and `VortexProgram` bind `libvortex.so`, manage buffers, upload kernels/args, launch with `vx_start_g`, and wait with `vx_ready_wait`.
- `extra/vortex/Makefile.kernel`: kernel-only Vortex build helper.
- `tinygrad/codegen/gpudims.py`: keeps local-dimension masks correct for Vortex global stores.
- `examples/vortex/`: example tinyvortex tensor programs

## Practical Notes

- Make sure your shell is not running with `DEBUG=release`; tinygrad parses `DEBUG` as an integer. Use `unset DEBUG` or `DEBUG=0` before running Python
- `LD_LIBRARY_PATH` must include the Vortex runtime directory before Python starts because Vortex's runtime stub dlopens driver libraries by name
- Config changes intentionally invalidate relevant tinyvortex compiler cache entries
- `trace/ramulator.log.*` is ignored by git; may be useful for debugging memory-system behavior

## Upcoming TODOs:
- Porting Vortex roofline perf plot
- Vortex MUFU intrinsics and tinygrad Uop mapping for accelerating activations/epilogue
- Trying out JITBEAM kernel search
- Run tinyvortex on AMD Xilinx U55C FPGA
