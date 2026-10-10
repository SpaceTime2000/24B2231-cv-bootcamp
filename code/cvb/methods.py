"""Training methods. A method owns: how to build its network(s), the training loss(es) for one batch, and how
to turn the network(s) into one set of test-time logits.

  SingleNet(loss)  - one ResNet-18 with any per-sample loss ('ce', 'ls', 'gce', 'sce')
  CoTeaching       - two ResNet-18s; each trains on the small-loss samples picked by the OTHER net
                     (Han et al., NeurIPS 2018). Counts as 2 networks, within the budget.

Data-level methods (filtering / relabeling with Task 3 scores) are not here: they are just SingleNet('ce')
trained on a modified (images, labels) set, see selection.py and task4_robust.py.
"""
import torch
import torch.nn.functional as F

from .losses import make_loss, ce


def _new_net(device, num_classes=10):
    from model import ResNet18  # provided architecture, untouched
    net = ResNet18(num_classes=num_classes).to(device)
    if device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)
    return net


class SingleNet:
    n_nets = 1

    def __init__(self, loss="ce", **loss_kw):
        self.loss_name, self.loss_kw = loss, loss_kw
        self.loss_fn = make_loss(loss, **loss_kw)

    def describe(self):
        return {"method": "single", "loss": self.loss_name, **self.loss_kw}

    def build(self, device):
        return [_new_net(device)]

    def step_losses(self, nets, x, y, epoch_float):
        return [self.loss_fn(nets[0](x).float(), y).mean()]

    def predict(self, nets, x):
        return nets[0](x).float()


class CoTeaching:
    """forget_rate = fraction of each batch treated as 'probably noisy' once ramped up. It follows the noise level
    stated in the problem statement (about 40%); it is NOT measured from clean labels. The kept fraction
    ramps linearly from 100% to (1 - forget_rate) over `ramp_epochs`."""
    n_nets = 2

    def __init__(self, forget_rate=0.4, ramp_epochs=10):
        self.forget_rate, self.ramp_epochs = forget_rate, ramp_epochs

    def describe(self):
        return {"method": "coteaching", "forget_rate": self.forget_rate, "ramp_epochs": self.ramp_epochs}

    def build(self, device):
        return [_new_net(device), _new_net(device)]

    def step_losses(self, nets, x, y, epoch_float):
        l1, l2 = ce(nets[0](x).float(), y), ce(nets[1](x).float(), y)
        r = self.forget_rate * min(epoch_float / self.ramp_epochs, 1.0)
        keep = max(1, int(round((1.0 - r) * len(y))))
        with torch.no_grad():
            i1 = torch.argsort(l1)[:keep]   # net 1's small-loss picks
            i2 = torch.argsort(l2)[:keep]   # net 2's small-loss picks
        return [l1[i2].mean(), l2[i1].mean()]  # each net learns from its peer's picks

    def predict(self, nets, x):
        p = sum(F.softmax(n(x).float(), 1) for n in nets) / len(nets)
        return p.clamp_min(1e-12).log()   # log-probs of the 2-net ensemble


def make_method(name: str, **kw):
    if name in ("ce", "ls", "gce", "sce"):
        return SingleNet(name, **kw)
    if name == "coteach":
        return CoTeaching(**kw)
    raise ValueError(f"unknown method {name}")
