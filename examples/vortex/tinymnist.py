import argparse

from tinygrad import Tensor, TinyJit, nn
from tinygrad.helpers import Context, trange
from tinygrad.nn.datasets import mnist


class TinyMNISTConvNet:
  def __init__(self, channels:int=8):
    self.l1 = nn.Conv2d(1, channels, kernel_size=(3, 3))
    self.l2 = nn.Conv2d(channels, channels*2, kernel_size=(3, 3))
    self.l3 = nn.Linear(channels*2*5*5, 10)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x.float() / 255.0).relu().max_pool2d((2, 2))
    x = self.l2(x).relu().max_pool2d((2, 2))
    return self.l3(x.flatten(1))


def main():
  parser = argparse.ArgumentParser(description="Train a tiny MNIST convnet on CPU, then run inference on VORTEX.")
  parser.add_argument("--steps", type=int, default=200, help="CPU training steps; increase this for better accuracy")
  parser.add_argument("--batch-size", type=int, default=256, help="CPU training batch size")
  parser.add_argument("--channels", type=int, default=16, help="Conv channels; lower is faster on the simulator")
  parser.add_argument("--eval-samples", type=int, default=1000, help="CPU test samples used for accuracy reporting")
  parser.add_argument("--infer-samples", type=int, default=1, help="Test samples to run through VORTEX")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")
  parser.add_argument("--skip-vortex", action="store_true", help="Only train and evaluate on CPU")
  args = parser.parse_args()

  Tensor.manual_seed(args.seed)

  # Keep the whole training path on CPU. The Vortex simulator is intentionally only used for inference below.
  with Context(DEV="CPU"):
    X_train, Y_train, X_test, Y_test = mnist(device="CPU")
    model = TinyMNISTConvNet(args.channels)
    opt = nn.optim.Adam(nn.state.get_parameters(model))

    @TinyJit
    @Context(TRAINING=1)
    def train_step() -> Tensor:
      opt.zero_grad()
      samples = Tensor.randint(args.batch_size, high=X_train.shape[0])
      loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
      return loss.realize(*opt.schedule_step())

    @TinyJit
    def test_acc(n:int) -> Tensor:
      return (model(X_test[:n]).argmax(axis=1) == Y_test[:n]).mean() * 100

    test_n = min(args.eval_samples, X_test.shape[0])
    acc = float("nan")
    for i in (t:=trange(args.steps)):
      loss = train_step()
      if i == args.steps-1 or i % 50 == 49: acc = test_acc(test_n).item()
      t.set_description(f"cpu loss: {loss.item():6.2f} cpu test_accuracy: {acc:5.2f}%")

    X_infer_cpu, Y_infer_cpu = X_test[:args.infer_samples].realize(), Y_test[:args.infer_samples].realize()
    with Context(TRAINING=0):
      cpu_logits = model(X_infer_cpu).realize()
    cpu_pred = cpu_logits.argmax(axis=1).realize()
    cpu_state = nn.state.get_state_dict(model)

  print(f"CPU training complete: test_accuracy={acc:.2f}% on {test_n} samples")
  if args.skip_vortex:
    print(f"CPU predictions: {cpu_pred.tolist()}")
    print(f"Labels:          {Y_infer_cpu.tolist()}")
    return

  # Create a matching model on VORTEX, copy the trained CPU weights to it, and run inference only.
  with Context(DEV="VORTEX"):
    vortex_model = TinyMNISTConvNet(args.channels)
    nn.state.load_state_dict(vortex_model, cpu_state, verbose=False)
    with Context(TRAINING=0):
      vortex_logits = vortex_model(X_infer_cpu.to("VORTEX")).realize().to("CPU").realize()

  vortex_pred = vortex_logits.argmax(axis=1).realize()
  err = (vortex_logits - cpu_logits).abs().max().item()
  print(f"VORTEX predictions: {vortex_pred.tolist()}")
  print(f"CPU predictions:    {cpu_pred.tolist()}")
  print(f"Labels:             {Y_infer_cpu.tolist()}")
  print(f"max |VORTEX-CPU logits| = {err:.6f}")
  if vortex_pred.tolist() == cpu_pred.tolist():
    print("PASSED!")
  else:
    print("FAILED! VORTEX predictions differ from CPU predictions")


if __name__ == "__main__":
  main()
