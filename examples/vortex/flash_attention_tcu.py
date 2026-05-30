import argparse, math

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import USE_TC
from tinygrad.runtime.ops_vortex import vortex_wmma_matmul

ITYPE_MAP = {
  "fp16": dtypes.half, "bf16": dtypes.bfloat16,
  "fp8": dtypes.fp8e4m3, "bf8": dtypes.fp8e5m2,
}
K_SCALE = {"fp16": 2, "bf16": 2, "fp8": 4, "bf8": 4}
TOL = {"fp16": 5e-2, "bf16": 1e-1, "fp8": 7e-1, "bf8": 1.0}

parser = argparse.ArgumentParser()
parser.add_argument("--itype", choices=ITYPE_MAP.keys(), default="fp16", help="TCU input format")
parser.add_argument("--seq", type=int, default=None, help="Sequence length, default is one KV block")
parser.add_argument("--heads", type=int, default=1, help="Number of attention heads")
parser.add_argument("--block", type=int, default=None, help="Q/KV block size, default is the current TCU tile multiple")
parser.add_argument("--base-dim", type=int, default=None, help="Base head dimension before format scaling")
parser.add_argument("--causal", action="store_true", help="Apply a lower-triangular causal mask")
args = parser.parse_args()

dtype_in = ITYPE_MAP[args.itype]


def require_multiple(name: str, value: int, multiple: int):
  if value % multiple: raise ValueError(f"{name} must be a multiple of {multiple}, got {value}")


def flash_attention_tcu(q: Tensor, k: Tensor, v: Tensor, block: int, causal: bool = False) -> Tensor:
  n, d = q.shape
  scale = 1.0 / math.sqrt(d)
  causal_mask = Tensor.ones(block, block, device="VORTEX", dtype=dtypes.bool).tril().realize() if causal else None
  k_blocks = [k[i:i+block].contiguous().realize() for i in range(0, n, block)]
  v_blocks = [v[i:i+block].T.contiguous().realize() for i in range(0, n, block)]
  outs = []

  for q0 in range(0, n, block):
    q_blk = q[q0:q0+block].contiguous().realize()
    m = Tensor.full((block, 1), -float("inf"), device="VORTEX", dtype=dtypes.float32).realize()
    l = Tensor.zeros(block, 1, device="VORTEX", dtype=dtypes.float32).realize()
    acc = Tensor.zeros(block, d, device="VORTEX", dtype=dtypes.float32).realize()
    stop = q0 + block if causal else n

    for k0 in range(0, stop, block):
      scores = (vortex_wmma_matmul(q_blk, k_blocks[k0 // block], dtype=dtypes.float, b_col_major=True) * scale).realize()
      if causal and k0 == q0: scores = causal_mask.where(scores, -float("inf")).realize()

      m_next = m.maximum(scores.max(axis=1, keepdim=True)).realize()
      old_scale = (m - m_next).exp().realize()
      p = (scores - m_next).exp().realize()

      pv = vortex_wmma_matmul(p.cast(dtype_in).contiguous(), v_blocks[k0 // block], dtype=dtypes.float, b_col_major=True)
      acc = (acc * old_scale + pv).realize()
      l = (l * old_scale + p.sum(axis=1, keepdim=True)).realize()
      m = m_next
    outs.append((acc / l).realize())
  return outs[0].cat(*outs[1:], dim=0).realize()


def stack_heads(outs: list[Tensor]) -> Tensor:
  return outs[0].stack(*outs[1:], dim=0).realize() if len(outs) > 1 else outs[0].unsqueeze(0).realize()


def multihead_flash_attention_tcu(q: Tensor, k: Tensor, v: Tensor, block: int, causal: bool = False) -> Tensor:
  return stack_heads([flash_attention_tcu(q[h], k[h], v[h], block, causal) for h in range(q.shape[0])])


renderer = Device["VORTEX"].renderer
if not USE_TC: raise RuntimeError("flash_attention_tcu.py requires tensor-core lowering; unset TC=0")
tc = next((tc for tc in renderer.tensor_cores if tc.dtype_in == dtype_in and tc.dtype_out == dtypes.float), None)
if tc is None:
  raise RuntimeError(f"no VORTEX TCU descriptor for itype={args.itype} otype=fp32; add -DVX_CFG_EXT_TCU_ENABLE to VORTEX_CONFIGS")

tile_n, tile_m, tile_k = tc.dims
block = args.block or max(tile_m, tile_n, tile_k)
seq = args.seq or block
heads = args.heads
dim = (args.base_dim if args.base_dim is not None else max(tile_k, tile_n) // K_SCALE[args.itype]) * K_SCALE[args.itype]

if heads < 1: raise ValueError(f"--heads must be >= 1, got {heads}")
require_multiple("--seq", seq, block)
require_multiple("--block", block, tile_m)
require_multiple("--block", block, tile_n)
require_multiple("--block", block, tile_k)
require_multiple("--base-dim scaled by input format", dim, tile_n)
require_multiple("--base-dim scaled by input format", dim, tile_k)

h_q = Tensor.rand(heads, seq, dim, device="CPU").cast(dtype_in).realize()
h_k = Tensor.rand(heads, seq, dim, device="CPU").cast(dtype_in).realize()
h_v = Tensor.rand(heads, seq, dim, device="CPU").cast(dtype_in).realize()

d_q, d_k, d_v = h_q.to("VORTEX"), h_k.to("VORTEX"), h_v.to("VORTEX")
d_o = multihead_flash_attention_tcu(d_q, d_k, d_v, block, args.causal)

h_o = d_o.to("CPU").realize()

refs = []
for h in range(heads):
  scores = h_q[h].cast(dtypes.float).matmul(h_k[h].cast(dtypes.float).T) * (1.0 / math.sqrt(dim))
  if args.causal: scores = Tensor.ones(seq, seq, device="CPU", dtype=dtypes.bool).tril().where(scores, -float("inf"))
  refs.append(scores.softmax(-1).cast(dtype_in).cast(dtypes.float).matmul(h_v[h].cast(dtypes.float), dtype=dtypes.float).realize())
h_o_ref = stack_heads(refs)
err = (h_o - h_o_ref).abs().max().item()
if err <= TOL[args.itype]:
  print("PASSED!")
else:
  print(f"FAILED! itype={args.itype} heads={heads} seq={seq} block={block} dim={dim} causal={args.causal} max error = {err}")
