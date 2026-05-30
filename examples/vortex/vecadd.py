import argparse

from tinygrad import Tensor

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1024, help="N dimension")
args = parser.parse_args()

N = args.n

# generate random values on CPU
h_a = Tensor.rand(N, device="CPU")
h_b = Tensor.rand(N, device="CPU")

# push data to VORTEX
d_a = h_a.to("VORTEX")
d_b = h_b.to("VORTEX")

# vecadd on VORTEX
d_c = (d_a + d_b).realize()

# pull GPU result back to CPU
h_c = d_c.to("CPU")

# verification

h_c_ref = h_a + h_b
err = (h_c - h_c_ref).abs().max().item()
if err < 1e-5:
  print("PASSED!")
else:
  print(f"FAILED! max error = {err}")
