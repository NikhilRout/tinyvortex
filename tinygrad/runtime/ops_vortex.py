from __future__ import annotations

import ctypes, functools, hashlib, os, pathlib, re, shlex, subprocess, tempfile, time
from dataclasses import dataclass
from typing import Any

from tinygrad.device import Buffer, Device
from tinygrad.device import BufferSpec, Compiled, Compiler, CompileError, LRUAllocator
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import ALLOW_TF32, DEBUG, DEV, getenv, mv_address, round_up, suppress_finalizing
from tinygrad.codegen.opt.tc import TensorCore
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, create_non_native_float_pats, pm_manual_bf16_cast, uops_to_dtypes, wmma_args
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

VX_MEM_READ_WRITE = 0x3
VX_MAX_TIMEOUT = 24 * 60 * 60 * 1000

VX_CAPS_NUM_THREADS = 0x1
VX_CAPS_NUM_WARPS = 0x2
VX_CAPS_NUM_CORES = 0x3
VX_CAPS_GLOBAL_MEM_SIZE = 0x5
VX_CAPS_LOCAL_MEM_SIZE = 0x6
VX_CAPS_ISA_FLAGS = 0x7
VX_CAPS_NUM_CLUSTERS = 0xA
VX_ISA_EXT_TCU = 1 << (32 + 9)


class VxQueueInfo(ctypes.Structure):
  _fields_ = [("struct_size", ctypes.c_size_t), ("next", ctypes.c_void_p), ("priority", ctypes.c_int), ("flags", ctypes.c_uint32)]


class VxLaunchInfo(ctypes.Structure):
  _fields_ = [("struct_size", ctypes.c_size_t), ("next", ctypes.c_void_p), ("kernel", ctypes.c_void_p),
              ("args_host", ctypes.c_void_p), ("args_size", ctypes.c_size_t), ("ndim", ctypes.c_uint32),
              ("grid_dim", ctypes.c_uint32 * 3), ("block_dim", ctypes.c_uint32 * 3), ("lmem_size", ctypes.c_uint32),
              ("cluster_dim", ctypes.c_uint32 * 3)]

_AXIS = "xyz"
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KERNEL_MAKEFILE = _REPO_ROOT / "extra" / "vortex" / "Makefile.kernel"
_VORTEX_TCU_THREADS = (4, 8, 16, 32)
_VORTEX_TCU_FORMATS = (
  (dtypes.half, dtypes.float, "fp16", "fp32"),
  (dtypes.bfloat16, dtypes.float, "bf16", "fp32"),
  (dtypes.fp8e4m3, dtypes.float, "fp8", "fp32"),
  (dtypes.fp8e5m2, dtypes.float, "bf8", "fp32"),
  (dtypes.float, dtypes.float, "tf32", "fp32"),
  (dtypes.int8, dtypes.int32, "int8", "int32"),
  (dtypes.uint8, dtypes.int32, "uint8", "int32"),
)


def _path_from_env(name: str) -> pathlib.Path | None:
  val = os.getenv(name)
  return pathlib.Path(val).expanduser().resolve() if val else None


def _parse_make_var(config_mk: pathlib.Path | None, name: str, default: str = "") -> str:
  if config_mk is None or not config_mk.exists(): return default
  pat = re.compile(rf"^\s*{re.escape(name)}\s*(?:\?|:)?=\s*(.*?)\s*$")
  for line in config_mk.read_text(errors="ignore").splitlines():
    if (m := pat.match(line)): return os.path.expandvars(m.group(1))
  return default


def _vortex_make_env() -> dict[str, str]:
  env = os.environ.copy()
  env.pop("DEBUG", None)
  if os.getenv("VORTEX_DEBUG"): env["DEBUG"] = os.getenv("VORTEX_DEBUG", "1")
  if os.getenv("VORTEX_PERF"):
    env["PERF"] = "1"
    env["VORTEX_PROFILING"] = os.getenv("VORTEX_PROFILING", os.getenv("VORTEX_PERF", "1"))
  return env


def _config_tokens(configs: str) -> tuple[str, ...]:
  try: return tuple(shlex.split(configs))
  except ValueError: return tuple(configs.split())


def _config_has_define(configs: str, name: str) -> bool:
  return any(tok == f"-D{name}" or tok.startswith(f"-D{name}=") for tok in _config_tokens(configs))


def _config_define_int(configs: str, name: str, default: int) -> int:
  for tok in _config_tokens(configs):
    if tok.startswith(f"-D{name}="): return int(tok.split("=", 1)[1], 0)
  return default


def _lg2(x: int) -> int:
  assert x > 0 and x & (x - 1) == 0
  return x.bit_length() - 1


def _vortex_tcu_geometry(num_threads: int, dtype_in: DType) -> tuple[tuple[int, int, int], tuple[int, int, int], int, int]:
  lg_tile_cap = _lg2(num_threads * 8)
  tile_n, tile_m = 1 << (lg_tile_cap // 2), 1 << (lg_tile_cap - (lg_tile_cap // 2))
  tile_k32 = (num_threads * 8) // max(tile_m, tile_n)
  lg_threads = _lg2(num_threads)
  tc_n, tc_m = 1 << (lg_threads // 2), 1 << (lg_threads - (lg_threads // 2))
  i_ratio = 4 // dtype_in.itemsize
  nra, nrb, nrc = (tile_m * tile_k32) // num_threads, (tile_n * tile_k32) // num_threads, (tile_m * tile_n) // num_threads
  return (tile_n, tile_m, tile_k32 * i_ratio), (nra * i_ratio, nrb * i_ratio, nrc), tc_n, tc_m


def _vortex_descriptor_swizzle(opts: tuple[str, ...], dims: tuple[int, int, int], elems: tuple[int, int, int]):
  locals_, upcasts, lcnt, ucnt = [], [], 0, 0
  zero_stride = [[], []]
  for opt in opts:
    if opt[0] == "l":
      name, lcnt = f"l{lcnt}", lcnt + 1
      locals_.append(name)
      zero_stride[int(opt[1])].append(name)
    else:
      name, ucnt = f"u{ucnt}", ucnt + 1
      upcasts.append(name)
      zero_stride[int(opt[1])].append(name)
  reduces = [f"r{i}" for i in range(_lg2(dims[2]))]
  all_names = locals_ + upcasts + reduces

  def make(which: int, elem_count: int):
    nonzero = [x for x in all_names if x not in zero_stride[which] and x[0] != "l"]
    tail = nonzero[:_lg2(elem_count)]
    tail += [x for x in all_names if x not in tail][:len(upcasts) + len(reduces) - len(tail)]
    local = tuple(x for x in all_names if x not in tail)
    return local, tuple(tail[:len(upcasts)]), tuple(tail[len(upcasts):])

  return make(0, elems[0]), make(1, elems[1])

@functools.cache
def _get_vortex_tensor_cores(num_threads: int) -> tuple[TensorCore, ...]:
  if num_threads not in _VORTEX_TCU_THREADS: return ()
  ret: list[TensorCore] = []
  for dtype_in, dtype_out, _, _ in _VORTEX_TCU_FORMATS:
    dims, elems, tc_n, tc_m = _vortex_tcu_geometry(num_threads, dtype_in)
    n_bits, m_bits, local_axes, upcast_axes = _lg2(dims[0]), _lg2(dims[1]), _lg2(num_threads), _lg2(elems[2])
    opts = ("l0",) * _lg2(tc_n) + ("l1",) * _lg2(tc_m) + ("u0",) * (n_bits - _lg2(tc_n)) + ("u1",) * (m_bits - _lg2(tc_m))

    swizzle = _vortex_descriptor_swizzle(opts, dims, elems)
    ret.append(TensorCore(dims, num_threads, elems, dtype_in, dtype_out, opts, swizzle))
  return tuple(ret)


def _vortex_input_fmt(dtype: DType) -> str:
  return {dtypes.half: "fp16", dtypes.bfloat16: "bf16", dtypes.fp8e4m3: "fp8", dtypes.fp8e5m2: "bf8",
          dtypes.float: "tf32", dtypes.int8: "int8", dtypes.uint8: "uint8"}[dtype.scalar()]


def _vortex_output_fmt(dtype: DType) -> str:
  return {dtypes.float: "fp32", dtypes.int32: "int32"}[dtype.scalar()]


def _vortex_wmma_api_kernel(m: int, n: int, k: int, threads: int, dtype_in: DType, dtype_out: DType) -> str:
  vt_in, vt_out = _vortex_input_fmt(dtype_in), _vortex_output_fmt(dtype_out)
  return f"""
#include <stdint.h>
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<{threads}, vt::{vt_in}, vt::{vt_out}>;

typedef struct tinyvortex_args_t {{
  uint64_t C_addr;
  uint64_t A_addr;
  uint64_t B_addr;
}} tinyvortex_args_t;

__kernel void kernel_main(tinyvortex_args_t* __UNIFORM__ arg) {{
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  constexpr uint32_t M = {m}u;
  constexpr uint32_t N = {n}u;
  constexpr uint32_t K = {k}u;

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  for (uint32_t i = 0; i < K; i += ctx::tileK) {{
    auto pTileA = pA + tile_row * K + i;
    auto pTileB = pB + tile_col * K + i;
    ctx::load_matrix_sync(fragA, pTileA, K);
    ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    ctx::mma_sync(fragC, fragA, fragB, fragC);
  }}

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}}
"""


@functools.cache
def _compile_vortex_wmma_api_kernel(cache_key: str, m: int, n: int, k: int, threads: int, dtype_in: DType, dtype_out: DType) -> bytes:
  return VortexCompiler(VortexConfig.from_target(DEV.target("VORTEX"))).compile(
    _vortex_wmma_api_kernel(m, n, k, threads, dtype_in, dtype_out))


def vortex_wmma_matmul(a, b, dtype: DType | None = None, b_col_major: bool = False):
  from tinygrad.tensor import Tensor
  from tinygrad.uop.ops import UOp

  if len(a.shape) != 2 or len(b.shape) != 2: raise ValueError("vortex_wmma_matmul expects 2D tensors")
  m, k = a.shape
  bk, n = (b.shape[1], b.shape[0]) if b_col_major else b.shape
  if k != bk: raise ValueError(f"matmul shape mismatch: A is {a.shape}, B is {b.shape}")
  dtype_in, dtype_out = a.dtype.scalar(), dtype or (dtypes.int32 if dtypes.is_int(a.dtype.scalar()) else dtypes.float)
  if b.dtype.scalar() != dtype_in: raise ValueError(f"A/B dtype mismatch: {a.dtype} vs {b.dtype}")

  cfg = VortexConfig.from_target(DEV.target("VORTEX"))
  threads = _config_define_int(cfg.configs, "VX_CFG_NUM_THREADS", 4)
  if not _config_has_define(cfg.configs, "VX_CFG_EXT_TCU_ENABLE"): raise RuntimeError("vortex_wmma_matmul requires -DVX_CFG_EXT_TCU_ENABLE")
  dims, _, _, _ = _vortex_tcu_geometry(threads, dtype_in)
  tile_n, tile_m, tile_k = dims
  if m % tile_m or n % tile_n or k % tile_k:
    raise ValueError(f"Vortex WMMA API path requires M,N,K multiples of {(tile_m, tile_n, tile_k)}, got {(m, n, k)}")

  dev = Device["VORTEX"]
  a_v = a.realize() if a.device == "VORTEX" else a.to("VORTEX").realize()
  b_v = b.realize() if b.device == "VORTEX" else b.to("VORTEX").realize()
  b_cm = b_v if b_col_major else b_v.T.contiguous().realize()
  out = Buffer("VORTEX", m * n, dtype_out).allocate()

  lib = _compile_vortex_wmma_api_kernel(cfg.cache_key, m, n, k, threads, dtype_in, dtype_out)
  prg = VortexProgram(dev, f"vortex_wmma_matmul_{m}_{n}_{k}", lib,
                      layout=(("buffer", "C_addr", 0), ("buffer", "A_addr", 1), ("buffer", "B_addr", 2)),
                      tcu_meta=(threads, ((_vortex_input_fmt(dtype_in), _vortex_output_fmt(dtype_out)),)))
  prg(out._buf, a_v._buffer()._buf, b_cm._buffer()._buf, global_size=(n // tile_n, m // tile_m, 1))
  return Tensor(UOp.from_buffer(out).reshape((m, n)))


@dataclass(frozen=True)
class VortexConfig:
  driver: str
  target: str
  home: pathlib.Path
  build: pathlib.Path | None
  path: pathlib.Path | None
  runtime_path: pathlib.Path
  kernel_path: pathlib.Path
  config_mk: pathlib.Path | None
  configs: str
  kernel_lib: str
  xlen: str
  arch: str

  @staticmethod
  def from_target(target) -> VortexConfig:
    path = _path_from_env("VORTEX_PATH")
    home = _path_from_env("VORTEX_HOME") or pathlib.Path.home() / "vortex"
    build = _path_from_env("VORTEX_BUILD") or home / "build"
    config_mk = build / "config.mk"
    if not config_mk.exists(): raise RuntimeError(f"missing Vortex build config: {config_mk}")

    xlen = _parse_make_var(config_mk, "XLEN", "32")
    runtime_path = path / "runtime" / "lib" if path is not None else build / "sw" / "runtime"
    kernel_path = path / "kernel" / f"lib{xlen}" if path is not None else build / "sw" / "kernel"
    if not runtime_path.exists(): raise RuntimeError(f"missing Vortex runtime path: {runtime_path}")
    if not kernel_path.exists(): raise RuntimeError(f"missing Vortex kernel path: {kernel_path}")

    iface = target.interface.upper()
    env_driver = os.getenv("VORTEX_DRIVER", "")
    if iface == "SIMX": driver, xrt_target = "simx", "xrtsim"
    elif iface == "RTLSIM": driver, xrt_target = "rtlsim", "xrtsim"
    elif iface in {"XRTSIM", "XRT"}: driver, xrt_target = "xrt", "xrtsim" if iface == "XRTSIM" else os.getenv("VORTEX_XRT_TARGET", "hw")
    elif env_driver: driver, xrt_target = env_driver, os.getenv("VORTEX_XRT_TARGET", "xrtsim")
    else: driver, xrt_target = "simx", "xrtsim"

    arch = f"xlen{xlen},{driver},configs={hashlib.sha256(os.getenv('VORTEX_CONFIGS', '').encode()).hexdigest()[:12]}"
    return VortexConfig(driver, xrt_target, home, build, path, runtime_path, kernel_path, config_mk,
                        os.getenv("VORTEX_CONFIGS", ""), os.getenv("VORTEX_KERNEL_LIB", "vortex2"), xlen, arch)

  @property
  def libvortex(self) -> pathlib.Path: return self.runtime_path / "libvortex.so"
  @property
  def driver_lib(self) -> pathlib.Path: return self.runtime_path / f"libvortex-{self.driver}.so"
  @property
  def cache_key(self) -> str:
    stamp = "|".join(str((p.stat().st_mtime_ns if p and p.exists() else 0)) for p in (self.config_mk, self.libvortex, self.driver_lib))
    raw = "|".join(map(str, [self.home, self.build, self.path, self.runtime_path, self.kernel_path, self.driver, self.target,
                             self.configs, self.kernel_lib, self.xlen, stamp, _KERNEL_MAKEFILE.stat().st_mtime_ns]))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]

  def ensure_runtime_built(self):
    if self.build is None or not (self.build / "sw" / "runtime").exists(): return
    if not (self.configs or os.getenv("VORTEX_PERF") or getenv("VORTEX_REBUILD", 0) or getenv("VORTEX_BUILD_RUNTIME", 0)): return
    target = {"simx": "simx", "rtlsim": "rtlsim", "xrt": "xrt"}.get(self.driver, self.driver)
    cmd = ["make", "-C", str(self.build / "sw" / "runtime"), "stub", target, f"CONFIGS={self.configs}", f"VORTEX_HOME={self.home}"]
    if self.driver == "xrt": cmd.append(f"TARGET={self.target}")
    cmd += shlex.split(os.getenv("VORTEX_MAKE_ARGS", ""))
    try:
      out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=_vortex_make_env()).decode()
      if DEBUG >= 1 and out: print(out)
    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"Vortex runtime build failed\ncommand: {' '.join(cmd)}\n{e.output.decode(errors='replace')}") from e


def _dtype_to_ctype(dt: DType):
  base = dt.scalar()
  return {
    dtypes.bool: ctypes.c_bool, dtypes.int8: ctypes.c_int8, dtypes.uint8: ctypes.c_uint8,
    dtypes.int16: ctypes.c_int16, dtypes.uint16: ctypes.c_uint16,
    dtypes.int32: ctypes.c_int32, dtypes.uint32: ctypes.c_uint32,
    dtypes.int64: ctypes.c_int64, dtypes.uint64: ctypes.c_uint64,
    dtypes.float: ctypes.c_float, dtypes.double: ctypes.c_double,
  }.get(base, ctypes.c_int32)


def _dtype_field_type(dt: DType) -> str:
  base = dt.scalar()
  return {
    dtypes.bool: "bool", dtypes.int8: "int8_t", dtypes.uint8: "uint8_t",
    dtypes.int16: "int16_t", dtypes.uint16: "uint16_t",
    dtypes.int32: "int32_t", dtypes.uint32: "uint32_t",
    dtypes.int64: "int64_t", dtypes.uint64: "uint64_t",
    dtypes.float: "float", dtypes.double: "double",
  }.get(base, "int32_t")


def _param_arg_from_name(name: str) -> int:
  if (m := re.match(r"data(\d+)(?:_|$)", name)) is None: raise RuntimeError(f"cannot infer Vortex buffer arg from {name!r}")
  return int(m.group(1))


def _render_wmma_call(ctx: VortexRenderer, x: UOp) -> str:
  raise RuntimeError("Vortex WMMA must render through the vx_tensor.h load_matrix_sync/mma_sync/store_matrix_sync API")


vortex_string_rewrite = PatternMatcher([
  (UPat(Ops.WMMA, name="x"), _render_wmma_call),
]) + base_rewrite

class VortexCompiler(Compiler):
  def __init__(self, config: VortexConfig):
    self.config = config
    super().__init__(f"compile_vortex_{config.cache_key}")

  def compile(self, src: str) -> bytes:
    if not _KERNEL_MAKEFILE.exists(): raise CompileError(f"missing Vortex kernel Makefile helper: {_KERNEL_MAKEFILE}")
    h = hashlib.sha256((src + self.config.cache_key).encode()).hexdigest()
    build_dir = pathlib.Path(tempfile.gettempdir()) / f"tinyvortex-{os.getuid()}" / h
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "kernel.cpp").write_text(src)
    if getenv("VORTEX_REBUILD", 0): subprocess.run(["make", "-f", str(_KERNEL_MAKEFILE), "clean"], cwd=build_dir, check=False)

    cmd = ["make", "-f", str(_KERNEL_MAKEFILE), "kernel.vxbin",
           f"VORTEX_HOME={self.config.home}", f"VORTEX_BUILD={self.config.build or ''}", f"VORTEX_PATH={self.config.path or ''}",
           f"VORTEX_RT_PATH={self.config.runtime_path}",
           f"VORTEX_KN_PATH={self.config.kernel_path}", f"KERNEL_LIB={self.config.kernel_lib}",
           f"CONFIGS={self.config.configs}", "VX_SRCS=kernel.cpp"]
    if self.config.driver == "xrt": cmd.append(f"TARGET={self.config.target}")
    cmd += shlex.split(os.getenv("VORTEX_MAKE_ARGS", ""))
    try:
      out = subprocess.check_output(cmd, cwd=build_dir, stderr=subprocess.STDOUT, env=_vortex_make_env()).decode()
      if DEBUG >= 4 and out: print(out)
      if DEBUG >= 7: subprocess.run(["make", "-f", str(_KERNEL_MAKEFILE), "kernel.dump"], cwd=build_dir, check=False)
      return (build_dir / "kernel.vxbin").read_bytes()
    except subprocess.CalledProcessError as e:
      raise CompileError(f"Vortex compile failed in {build_dir}\ncommand: {' '.join(cmd)}\n{e.output.decode(errors='replace')}") from e

  def disassemble(self, lib: bytes):
    if DEBUG < 7: return
    h = hashlib.sha256(lib).hexdigest()
    out = pathlib.Path(tempfile.gettempdir()) / f"tinyvortex-{os.getuid()}" / f"{h}.vxbin"
    out.write_bytes(lib)
    print(f"Vortex vxbin: {out}")


class VortexRenderer(CStyleLanguage):
  has_aux = True
  has_threads = False
  has_local = True
  supports_float4 = False
  shared_max = 32768
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  barrier = "__syncthreads();"
  smem_prefix_for_cast = False
  float4 = "(float4)"
  float4_style = ("{", "}")
  gep_arr_threshold = 0
  type_map = {
    dtypes.bool: "bool", dtypes.int8: "int8_t", dtypes.uint8: "uint8_t",
    dtypes.int16: "int16_t", dtypes.uint16: "uint16_t", dtypes.int32: "int32_t",
    dtypes.uint32: "uint32_t", dtypes.int64: "int64_t", dtypes.uint64: "uint64_t",
    dtypes.half: "uint16_t", dtypes.bfloat16: "uint16_t", dtypes.fp8e4m3: "uint8_t", dtypes.fp8e5m2: "uint8_t",
  }
  code_for_workitem = {
    "g": lambda x: f"blockIdx.{_AXIS[int(x)]}",
    "l": lambda x: f"threadIdx.{_AXIS[int(x)]}",
    "i": lambda x: f"(blockIdx.{_AXIS[int(x)]}*blockDim.{_AXIS[int(x)]}+threadIdx.{_AXIS[int(x)]})",
  }
  code_for_op = {**CStyleLanguage.code_for_op,
    Ops.FDIV: lambda a,b,dtype: f"({a}/{b})",
    Ops.MAX: lambda a,b,dtype: f"tinyvortex_max({a},{b})",
    Ops.MULACC: lambda a,b,c,dtype: f"(({a}*{b})+{c})"}
  string_rewrite = vortex_string_rewrite
  extra_matcher = create_non_native_float_pats((dtypes.bfloat16, *dtypes.fp8s), casting=False) + pm_manual_bf16_cast + \
    (CStyleLanguage.extra_matcher or PatternMatcher([]))

  def __init__(self, target):
    super().__init__(target)
    self.config = VortexConfig.from_target(target)
    self.compiler = VortexCompiler(self.config)
    num_threads = _config_define_int(self.config.configs, "VX_CFG_NUM_THREADS", 4)
    num_warps = _config_define_int(self.config.configs, "VX_CFG_NUM_WARPS", 4)
    self.local_max = (num_threads * num_warps, 1, 1)
    self.tensor_cores = list(_get_vortex_tensor_cores(num_threads)) \
      if _config_has_define(self.config.configs, "VX_CFG_EXT_TCU_ENABLE") else []
    if not ALLOW_TF32: self.tensor_cores = [tc for tc in self.tensor_cores if tc.dtype_in != dtypes.float]
    self._last_aux: tuple[Any, ...] = ((), 0, None)
    self._lmem_size = 0

  def render(self, uops: list[UOp]) -> str:
    self._lmem_size = 0
    if any(u.op is Ops.WMMA for u in uops):
      raise RuntimeError("Vortex WMMA must use vortex_wmma_matmul so B is materialized column-major before load_matrix_sync")
    return self.render_kernel(*self._render(uops), uops)

  def render_buffer(self, x: UOp) -> str:
    if x.addrspace != AddrSpace.LOCAL: return super().render_buffer(x)
    base, count = x.dtype.scalar(), x.max_numel()
    align = min(max(base.itemsize, 4), 16)
    off = round_up(self._lmem_size, align)
    self._lmem_size = off + count * base.itemsize
    ctype = self.render_dtype(base)
    return f"{ctype}* {self[x]} = ({ctype}*)((uint8_t*)__local_mem() + {off});"

  def render_kernel(self, function_name: str, kernel: list[str], bufs: list[tuple[str, tuple[UOp|DType, bool]]], uops: list[UOp], prefix=None) -> str:
    layout = []
    arg_decls, call_args, fields = [], [], []
    for name, (buf, mutable) in bufs:
      dtype = buf.dtype if isinstance(buf, UOp) else buf
      is_buffer = isinstance(buf, UOp) and buf.addrspace == AddrSpace.GLOBAL
      if is_buffer:
        ctype = self._render_dtype(dtype, addrspace=AddrSpace.GLOBAL, mutable=mutable) if isinstance(buf, UOp) else self.render_dtype(dtype, mutable)
        arg_decls.append(f"{ctype} {name}")
        fields.append(f"  uint64_t {name};")
        call_args.append(f"reinterpret_cast<{ctype}>(arg->{name})")
        layout.append(("buffer", name, _param_arg_from_name(name)))
      else:
        ctype = self.render_dtype(dtype)
        arg_decls.append(f"{ctype} {name}")
        fields.append(f"  {_dtype_field_type(dtype)} {name};")
        call_args.append(f"arg->{name}")
        layout.append(("scalar", name, dtype))

    vec_prefix = [self.render_vector_prefix(dt) for dt in uops_to_dtypes(uops) if dt.count > 1]
    wmma_helpers = self._render_wmma_helpers(uops)
    helpers = [
      "#include <stdint.h>",
      "#include <math.h>",
      "#include <vx_spawn2.h>",
      "#include <vx_intrinsics.h>",
      "template <typename T> static inline T tinyvortex_max(T a, T b) { return a > b ? a : b; }",
      "static inline float tinyvortex_max(float a, float b) { return fmaxf(a, b); }",
      "static inline double tinyvortex_max(double a, double b) { return fmax(a, b); }",
      *vec_prefix,
    ]
    used_ops = {u.op for u in uops}
    if Ops.WMMA in used_ops: helpers += ["#include <vx_tensor.h>", "namespace vt = vortex::tensor;"]
    helpers += wmma_helpers
    if prefix: helpers.extend(prefix)

    src = [
      *helpers,
      "typedef struct tinyvortex_args_t {",
      *fields,
      "} tinyvortex_args_t;",
      f"static inline void {function_name}({', '.join(arg_decls)}) {{",
      *kernel,
      "}",
      "__kernel void kernel_main(tinyvortex_args_t* __UNIFORM__ arg) {",
      f"  {function_name}({', '.join(call_args)});",
      "}",
    ]
    tcu_threads = sorted({x[5] for x in wmma_args(uops)})
    tcu_meta = None if not tcu_threads else (
      tcu_threads[0],
      tuple(sorted({(_vortex_input_fmt(x[2]), _vortex_output_fmt(x[3])) for x in wmma_args(uops)})),
    )
    self._last_aux = (tuple(layout), self._lmem_size, tcu_meta)
    return "\n".join(src)

  def render_vector_prefix(self, dt: DType) -> str:
    scal, vec = self.render_dtype(dt.scalar()), self.render_dtype(dt)
    return f"typedef {scal} {vec} __attribute__((aligned({dt.itemsize}),ext_vector_type({dt.count})));"

  def _render_wmma_helpers(self, uops: list[UOp]) -> list[str]:
    return []

  def aux(self, uops: list[UOp]): return self._last_aux

  def supported_dtypes(self):
    return {d for d in super().supported_dtypes() if d in {
      dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16,
      dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64, dtypes.float,
      dtypes.bfloat16}}


@dataclass
class VortexBuffer:
  handle: ctypes.c_void_p
  addr: int
  size: int
  offset: int = 0
  owner: bool = True


class VortexRuntime:
  _instance: VortexRuntime | None = None

  def __init__(self, config: VortexConfig):
    self.config = config
    config.ensure_runtime_built()
    if os.getenv("VORTEX_PERF"):
      os.environ["VORTEX_PROFILING"] = os.getenv("VORTEX_PROFILING", os.getenv("VORTEX_PERF", "1"))
    if not config.libvortex.exists(): raise RuntimeError(f"missing Vortex runtime stub: {config.libvortex}")
    os.environ["VORTEX_DRIVER"] = config.driver
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{config.runtime_path}{':' + old_ld if old_ld else ''}"
    try:
      self.lib = ctypes.CDLL(str(config.libvortex), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
      raise RuntimeError(f"failed to load {config.libvortex}: {e}") from e
    self._bind()
    self.dev = ctypes.c_void_p()
    err = self.lib.vx_dev_open(ctypes.byref(self.dev))
    if err != 0:
      raise RuntimeError(
        f"vx_dev_open failed for VORTEX_DRIVER={config.driver}. If the driver library exists at {config.driver_lib}, "
        f"start Python with LD_LIBRARY_PATH={config.runtime_path}:$LD_LIBRARY_PATH so Vortex's stub dlopen can find it.")
    self.queue = ctypes.c_void_p()
    qinfo = VxQueueInfo(ctypes.sizeof(VxQueueInfo), None, 1, 0)
    self.check(self.lib.vx_queue_create(self.dev, ctypes.byref(qinfo), ctypes.byref(self.queue)), "vx_queue_create")

  @classmethod
  def open(cls, config: VortexConfig) -> VortexRuntime:
    if cls._instance is not None and cls._instance.config.driver != config.driver:
      raise RuntimeError("only one Vortex driver can be active per process; start a fresh process to switch drivers")
    if cls._instance is None: cls._instance = VortexRuntime(config)
    return cls._instance

  def _bind(self):
    v, vp, u32p, u64p = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64)
    self.lib.vx_dev_open.argtypes, self.lib.vx_dev_open.restype = [vp], ctypes.c_int
    self.lib.vx_dev_close.argtypes, self.lib.vx_dev_close.restype = [v], ctypes.c_int
    self.lib.vx_dev_caps.argtypes, self.lib.vx_dev_caps.restype = [v, ctypes.c_uint32, u64p], ctypes.c_int
    self.lib.vx_mem_alloc.argtypes, self.lib.vx_mem_alloc.restype = [v, ctypes.c_uint64, ctypes.c_int, vp], ctypes.c_int
    self.lib.vx_mem_free.argtypes, self.lib.vx_mem_free.restype = [v], ctypes.c_int
    self.lib.vx_mem_address.argtypes, self.lib.vx_mem_address.restype = [v, u64p], ctypes.c_int
    self.lib.vx_copy_to_dev.argtypes, self.lib.vx_copy_to_dev.restype = [v, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64], ctypes.c_int
    self.lib.vx_copy_from_dev.argtypes, self.lib.vx_copy_from_dev.restype = [ctypes.c_void_p, v, ctypes.c_uint64, ctypes.c_uint64], ctypes.c_int
    self.lib.vx_upload_kernel_bytes.argtypes, self.lib.vx_upload_kernel_bytes.restype = [v, ctypes.c_void_p, ctypes.c_uint64, vp], ctypes.c_int
    self.lib.vx_upload_bytes.argtypes, self.lib.vx_upload_bytes.restype = [v, ctypes.c_void_p, ctypes.c_uint64, vp], ctypes.c_int
    self.lib.vx_module_load_bytes.argtypes, self.lib.vx_module_load_bytes.restype = [v, ctypes.c_void_p, ctypes.c_size_t, vp], ctypes.c_int
    self.lib.vx_module_get_kernel.argtypes, self.lib.vx_module_get_kernel.restype = [v, ctypes.c_char_p, vp], ctypes.c_int
    self.lib.vx_module_release.argtypes, self.lib.vx_module_release.restype = [v], ctypes.c_int
    self.lib.vx_kernel_release.argtypes, self.lib.vx_kernel_release.restype = [v], ctypes.c_int
    self.lib.vx_device_dump_perf.argtypes, self.lib.vx_device_dump_perf.restype = [v, ctypes.c_void_p], ctypes.c_int
    self.lib.vx_queue_create.argtypes, self.lib.vx_queue_create.restype = [v, ctypes.POINTER(VxQueueInfo), vp], ctypes.c_int
    self.lib.vx_queue_release.argtypes, self.lib.vx_queue_release.restype = [v], ctypes.c_int
    self.lib.vx_enqueue_launch.argtypes, self.lib.vx_enqueue_launch.restype = [v, ctypes.POINTER(VxLaunchInfo), ctypes.c_uint32, vp, vp], ctypes.c_int
    self.lib.vx_event_wait_value.argtypes, self.lib.vx_event_wait_value.restype = [v, ctypes.c_uint64, ctypes.c_uint64], ctypes.c_int
    self.lib.vx_event_release.argtypes, self.lib.vx_event_release.restype = [v], ctypes.c_int
    self.lib.vx_max_occupancy_grid.argtypes = [v, ctypes.c_uint32, u32p, u32p, u32p]
    self.lib.vx_max_occupancy_grid.restype = ctypes.c_int
    self.lib.vx_ready_wait.argtypes, self.lib.vx_ready_wait.restype = [v, ctypes.c_uint64], ctypes.c_int

  def check(self, err: int, op: str):
    if err != 0: raise RuntimeError(f"Vortex runtime error {err} from {op}")

  def caps(self, cap: int) -> int:
    out = ctypes.c_uint64()
    self.check(self.lib.vx_dev_caps(self.dev, cap, ctypes.byref(out)), "vx_dev_caps")
    return out.value

  def close(self):
    if getattr(self, "dev", None):
      if getattr(self, "queue", None): self.lib.vx_queue_release(self.queue)
      self.lib.vx_dev_close(self.dev)
      self.dev = None


class VortexAllocator(LRUAllocator["VortexDevice"]):
  def _alloc(self, size: int, options: BufferSpec) -> VortexBuffer:
    if options.external_ptr is not None: raise RuntimeError("VORTEX does not support external_ptr buffers yet")
    handle = ctypes.c_void_p()
    self.dev.rt.check(self.dev.rt.lib.vx_mem_alloc(self.dev.rt.dev, size, VX_MEM_READ_WRITE, ctypes.byref(handle)), "vx_mem_alloc")
    addr = ctypes.c_uint64()
    self.dev.rt.check(self.dev.rt.lib.vx_mem_address(handle, ctypes.byref(addr)), "vx_mem_address")
    return VortexBuffer(handle, addr.value, size)

  @suppress_finalizing
  def _free(self, opaque: VortexBuffer, options: BufferSpec):
    if opaque.owner: self.dev.rt.check(self.dev.rt.lib.vx_mem_free(opaque.handle), "vx_mem_free")

  def _copyin(self, dest: VortexBuffer, src: memoryview):
    buf = ctypes.create_string_buffer(src.tobytes())
    self.dev.rt.check(self.dev.rt.lib.vx_copy_to_dev(dest.handle, ctypes.cast(buf, ctypes.c_void_p), dest.offset, len(src)), "vx_copy_to_dev")

  def _copyout(self, dest: memoryview, src: VortexBuffer):
    self.dev.rt.check(self.dev.rt.lib.vx_copy_from_dev(ctypes.c_void_p(mv_address(dest)), src.handle, src.offset, len(dest)), "vx_copy_from_dev")

  def _offset(self, buf: VortexBuffer, size: int, offset: int):
    return VortexBuffer(buf.handle, buf.addr + offset, size, buf.offset + offset, owner=False)


class VortexProgram:
  def __init__(self, dev: VortexDevice, name: str, lib: bytes, layout=(), lmem_size: int = 0, tcu_meta=None,
               runtimevars=None, prg: UOp | None = None, **kwargs):
    self.dev, self.name, self.lib, self.layout, self.lmem_size, self.prg = dev, name, lib, tuple(layout), int(lmem_size), prg
    self.tcu_meta, self._tcu_checked = tcu_meta, False
    self.var_names = [v.arg[0] for v in (prg.arg.vars if prg is not None else ())]
    self.global_args = tuple(prg.arg.globals) if prg is not None else tuple(i for i,e in enumerate(self.layout) if e[0] == "buffer")

  def _pack_args(self, bufs: tuple[VortexBuffer, ...], vals: tuple[int, ...]):
    fields = []
    values = []
    global_pos = {arg: i for i, arg in enumerate(self.global_args)}
    val_pos = {name: i for i, name in enumerate(self.var_names)}
    for ent in self.layout:
      if ent[0] == "buffer":
        _, name, arg = ent
        b = bufs[global_pos[arg]]
        fields.append((name, ctypes.c_uint64))
        values.append(b.addr)
      else:
        _, name, dt = ent
        fields.append((name, _dtype_to_ctype(dt)))
        v = vals[val_pos[name]] if name in val_pos and vals[val_pos[name]] is not None else 0
        values.append(v)
    Args = type("VortexArgs", (ctypes.Structure,), {"_fields_": fields})
    return Args(*values)

  def _check_tcu_caps(self):
    if self.tcu_meta is None or self._tcu_checked: return
    tcu_threads = self.tcu_meta[0]
    isa_flags, threads = self.dev.rt.caps(VX_CAPS_ISA_FLAGS), self.dev.rt.caps(VX_CAPS_NUM_THREADS)
    if (isa_flags & VX_ISA_EXT_TCU) == 0:
      raise RuntimeError("Vortex tensor-core kernel requires runtime built with -DVX_CFG_EXT_TCU_ENABLE")
    if threads != tcu_threads:
      raise RuntimeError(f"Vortex tensor-core kernel was compiled for VX_CFG_NUM_THREADS={tcu_threads} but runtime reports {threads}")
    self._tcu_checked = True

  def __call__(self, *bufs: VortexBuffer, global_size: tuple[int, ...] = (1, 1, 1), local_size: tuple[int, ...] | None = None,
               vals: tuple[int, ...] = (), wait=False, timeout=None, **kw) -> float | None:
    self._check_tcu_caps()
    raw = ctypes.create_string_buffer(self.lib)
    module, kernel = ctypes.c_void_p(), ctypes.c_void_p()
    self.dev.rt.check(self.dev.rt.lib.vx_module_load_bytes(self.dev.rt.dev, ctypes.cast(raw, ctypes.c_void_p), len(self.lib),
                                                           ctypes.byref(module)), "vx_module_load_bytes")
    self.dev.rt.check(self.dev.rt.lib.vx_module_get_kernel(module, b"main", ctypes.byref(kernel)), "vx_module_get_kernel")
    args = self._pack_args(bufs, vals)
    try:
      ndim = len(global_size)
      Grid = ctypes.c_uint32 * ndim
      grid_dim, block_dim = Grid(), Grid()
      if local_size is None and self.tcu_meta is not None:
        api_grid = self.tcu_meta[2] if len(self.tcu_meta) > 2 else None
        launch_size = api_grid or global_size
        ndim = len(launch_size)
        Grid = ctypes.c_uint32 * ndim
        grid_dim, block_dim = Grid(*[int(x) for x in launch_size]), Grid(*([int(self.tcu_meta[0])] + [1] * (ndim - 1)))
      elif local_size is None:
        gdim = Grid(*[int(x) for x in global_size])
        self.dev.rt.check(self.dev.rt.lib.vx_max_occupancy_grid(self.dev.rt.dev, ndim, gdim, grid_dim, block_dim), "vx_max_occupancy_grid")
      else:
        grid_dim, block_dim = Grid(*[int(x) for x in global_size]), Grid(*[int(x) for x in local_size])
      launch = VxLaunchInfo()
      launch.struct_size, launch.kernel = ctypes.sizeof(VxLaunchInfo), kernel
      launch.args_host, launch.args_size, launch.ndim = ctypes.cast(ctypes.byref(args), ctypes.c_void_p), ctypes.sizeof(args), ndim
      for i in range(3):
        launch.grid_dim[i] = grid_dim[i] if i < ndim else 1
        launch.block_dim[i] = block_dim[i] if i < ndim else 1
        launch.cluster_dim[i] = 1
      launch.lmem_size = self.lmem_size
      event = ctypes.c_void_p()
      st = time.perf_counter()
      self.dev.rt.check(self.dev.rt.lib.vx_enqueue_launch(self.dev.rt.queue, ctypes.byref(launch), 0, None, ctypes.byref(event)), "vx_enqueue_launch")
      timeout_ns = int(timeout or getenv("VORTEX_TIMEOUT", VX_MAX_TIMEOUT)) * 1_000_000
      self.dev.rt.check(self.dev.rt.lib.vx_event_wait_value(event, 1, timeout_ns), "vx_event_wait_value")
      self.dev.rt.lib.vx_event_release(event)
      self.dev.rt.check(self.dev.rt.lib.vx_device_dump_perf(self.dev.rt.dev, None), "vx_device_dump_perf")
    finally:
      if kernel: self.dev.rt.lib.vx_kernel_release(kernel)
      if module: self.dev.rt.lib.vx_module_release(module)
    return time.perf_counter() - st if wait else None


class VortexDevice(Compiled):
  def __init__(self, device: str = "VORTEX"):
    self.config = VortexConfig.from_target(DEV.target("VORTEX"))
    self.rt = VortexRuntime.open(self.config)
    super().__init__(device, VortexAllocator(self), [VortexRenderer], functools.partial(VortexProgram, self), arch=self.config.arch)

  def synchronize(self): pass

  def finalize(self):
    if hasattr(self, "allocator"): self.allocator.free_cache()
    if VortexRuntime._instance is self.rt:
      self.rt.close()
      VortexRuntime._instance = None

  def count(self) -> int: return 1

  def caps(self) -> dict[str, int]:
    return {
      "threads": self.rt.caps(VX_CAPS_NUM_THREADS),
      "warps": self.rt.caps(VX_CAPS_NUM_WARPS),
      "cores": self.rt.caps(VX_CAPS_NUM_CORES),
      "clusters": self.rt.caps(VX_CAPS_NUM_CLUSTERS),
      "global_mem": self.rt.caps(VX_CAPS_GLOBAL_MEM_SIZE),
      "local_mem": self.rt.caps(VX_CAPS_LOCAL_MEM_SIZE),
      "isa_flags": self.rt.caps(VX_CAPS_ISA_FLAGS),
    }
