import argparse

from tinygrad import Tensor

parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, default=256, help="M dimension")
parser.add_argument("--n", type=int, default=256, help="N dimension")
parser.add_argument("--k", type=int, default=256, help="K dimension")
args = parser.parse_args()

M, N, K = args.m, args.n, args.k

# generate random values on CPU
h_a = Tensor.rand(M, K, device="CPU")
h_b = Tensor.rand(K, N, device="CPU")

# push data to VORTEX
d_a = h_a.to("VORTEX")
d_b = h_b.to("VORTEX")

# sgemm on VORTEX
d_c = d_a.matmul(d_b).realize()

# pull GPU result back to CPU
h_c = d_c.to("CPU")

# verification
h_c_ref = h_a.matmul(h_b)
err = (h_c - h_c_ref).abs().max().item()
if err < 1e-3:
  print("PASSED!")
else:
  print(f"FAILED! max error = {err}")
