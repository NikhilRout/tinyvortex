import argparse
from contextlib import nullcontext

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import Context, USE_TC
from tinygrad.runtime.ops_vortex import vortex_wmma_matmul

ITYPE_MAP = {
  "fp16": dtypes.half, "bf16": dtypes.bfloat16,
  "fp8": dtypes.fp8e4m3, "bf8": dtypes.fp8e5m2,
  "tf32": dtypes.float,
  "int8": dtypes.int8,
}
OTYPE_MAP = {"fp32": dtypes.float, "int32": dtypes.int32}
FP_ITYPES = {"fp16", "bf16", "fp8", "bf8", "tf32"}
INT_ITYPES = {"int8"}
K_SCALE = {"fp16": 2, "bf16": 2, "fp8": 4, "bf8": 4, "tf32": 1, "int8": 4}
TOL = {"fp16": 2e-1, "bf16": 4e-1, "fp8": 4.0, "bf8": 8.0, "tf32": 5e-1, "int8": 0.0}

parser = argparse.ArgumentParser()
parser.add_argument("--itype", choices=ITYPE_MAP.keys(), default="fp16", help="Input format")
parser.add_argument("--otype", choices=OTYPE_MAP.keys(), default="fp32", help="Output/accumulator format")
parser.add_argument("--m", type=int, default=256, help="M dimension")
parser.add_argument("--n", type=int, default=256, help="N dimension")
parser.add_argument("--base-k", type=int, default=256, help="Base K dimension before format scaling")
args = parser.parse_args()

M, N, K = args.m, args.n, args.base_k * K_SCALE[args.itype]
dtype_in, dtype_out = ITYPE_MAP[args.itype], OTYPE_MAP[args.otype]
if args.itype in FP_ITYPES and args.otype != "fp32": raise ValueError(f"{args.itype} requires --otype=fp32")
if args.itype in INT_ITYPES and args.otype != "int32": raise ValueError(f"{args.itype} requires --otype=int32")

# generate random values on CPU
if args.itype in INT_ITYPES:
  h_a = Tensor.randint(M, K, device="CPU", dtype=dtype_in).realize()
  h_b = Tensor.randint(K, N, device="CPU", dtype=dtype_in).realize()
else:
  h_a = Tensor.rand(M, K, device="CPU").cast(dtype_in).realize()
  h_b = Tensor.rand(K, N, device="CPU").cast(dtype_in).realize()

ctx = Context(ALLOW_TF32=1) if args.itype == "tf32" else nullcontext()
with ctx:
  renderer = Device["VORTEX"].renderer
  if not USE_TC: raise RuntimeError("sgemm_tcu.py requires tensor-core lowering; unset TC=0")
  if not any(tc.dtype_in == dtype_in and tc.dtype_out == dtype_out for tc in renderer.tensor_cores):
    raise RuntimeError(f"no VORTEX TCU descriptor for itype={args.itype} otype={args.otype}; add -DVX_CFG_EXT_TCU_ENABLE to VORTEX_CONFIGS")

  # push data to VORTEX
  d_a = h_a.to("VORTEX")
  d_b_col = h_b.T.contiguous().to("VORTEX")

  # sgemm on VORTEX tensor cores through the Vortex WMMA load/mma/store API
  d_c = vortex_wmma_matmul(d_a, d_b_col, dtype=dtype_out, b_col_major=True).realize()

# pull GPU result back to CPU
h_c = d_c.to("CPU").realize()

# verification
h_c_ref = h_a.cast(dtype_out).matmul(h_b.cast(dtype_out), dtype=dtype_out)
err = (h_c.cast(dtype_out) - h_c_ref).abs().max().item()
if err <= TOL[args.itype]:
  print("PASSED!")
else:
  print(f"FAILED! itype={args.itype} otype={args.otype} M={M} N={N} K={K} max error = {err}")
