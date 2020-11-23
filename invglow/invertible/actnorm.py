import torch as th
from torch import nn
import numpy as np

def inverse_elu(y):
    mask = y > 1
    x = th.zeros_like(y)
    x.data[mask] = y.data[mask] - 1
    x.data[1-mask] = th.log(y.data[1-mask])
    return x

class ActNorm(nn.Module):
    def __init__(self, in_channel, scale_fn, eps=1e-8, verbose_init=True,
                 init_eps=None):
        super().__init__()

        self.loc = nn.Parameter(th.zeros(1, in_channel, 1, 1))
        self.log_scale = nn.Parameter(th.zeros(1, in_channel, 1, 1))

        self.initialize_this_forward = False
        self.initialized = False
        self.scale_fn = scale_fn
        self.eps = eps
        self.verbose_init = verbose_init
        if init_eps is None:
            if scale_fn == 'exp':
                self.init_eps = 1e-6
            else:
                assert scale_fn == 'elu'
                self.init_eps = 1e-1
        else:
            self.init_eps = init_eps

    def initialize(self, x):
        with th.no_grad():
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            self.loc.data.copy_(-mean)
            if self.scale_fn == 'exp':
                self.log_scale.data.copy_(th.log(1 / th.clamp_min(std, self.init_eps)))
            elif self.scale_fn == 'elu':
                self.log_scale.data.copy_(inverse_elu(1 / th.clamp_min(std, self.init_eps)))
            else:
                assert False

            if self.scale_fn == 'exp':
                multipliers = th.exp(self.log_scale.squeeze())
            elif self.scale_fn == 'elu':
                multipliers = th.nn.functional.elu(self.log_scale) + 1
            if self.verbose_init:
                print(f"Multiplier init to (log10) "
                f"min: {np.log10(th.min(multipliers).item()):3.0f} "
                f"max: {np.log10(th.max(multipliers).item()):3.0f} "
                f"mean: {np.log10(th.mean(multipliers).item()):3.0f}")

    def forward(self, x, fixed=None):
        was_2d = False
        if len (x.shape) == 2:
            was_2d = True
            x = x.unsqueeze(-1).unsqueeze(-1)
        _, _, height, width = x.shape

        if not self.initialized:
            assert self.initialize_this_forward, (
                "Please first initialize by setting initialize_this_forward to True"
                " and forwarding appropriate data")
        if self.initialize_this_forward:
            self.initialize(x)
            self.initialized = True
            self.initialize_this_forward = False

        scale, log_det_px = self.scale_and_logdet_per_pixel()
        y = scale * (x + self.loc)
        if was_2d:
            y = y.squeeze(-1).squeeze(-1)

        logdet = height * width * log_det_px
        logdet = logdet.repeat(len(
            x))

        return y, logdet

    def scale_and_logdet_per_pixel(self):
        if self.scale_fn == 'exp':
            scale = th.exp(self.log_scale) + self.eps
            if self.eps == 0:
                logdet = th.sum(self.log_scale)
            else:
                logdet = th.sum(th.log(scale))
        elif self.scale_fn == 'elu':
            scale = th.nn.functional.elu(self.log_scale) + 1 + self.eps
            logdet = th.sum(th.log(scale))
        else:
            assert False

        return scale, logdet

    def invert(self, y, fixed=None):
        was_2d = False
        if len (y.shape) == 2:
            was_2d = True
            y = y.unsqueeze(-1).unsqueeze(-1)
        _, _, height, width = y.shape
        scale, log_det_px = self.scale_and_logdet_per_pixel()
        x = y / scale - self.loc
        logdet = height * width * log_det_px
        if was_2d:
            x = x.squeeze(-1).squeeze(-1)
        # repeat per example in batch
        logdet = logdet.repeat(len(
            x))
        return x, logdet


def init_act_norm(net, trainloader, n_batches=10, uni_noise_factor=1/255.0):
    if trainloader is not None:
        all_x = []
        for i_batch, (x, y) in enumerate(trainloader):
            all_x.append(x)
            if i_batch >= n_batches:
                break

        init_x = th.cat(all_x, dim=0)
        init_x = init_x.cuda()
        init_x = init_x + th.rand_like(init_x) * uni_noise_factor

        for m in net.modules():
            if hasattr(m, 'initialize_this_forward'):
                m.initialize_this_forward = True

        _ = net(init_x)
    else:
        for m in net.modules():
            if hasattr(m, 'initialize_this_forward'):
                m.initialized = True


class PureActNorm(nn.Module):
    def __init__(self, in_channel,):
        super().__init__()
        self.loc = nn.Parameter(th.zeros(in_channel))
        self.scale = nn.Parameter(th.zeros(in_channel))
        self.initialize_this_forward = False
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            assert self.initialize_this_forward, (
                "Please first initialize by setting initialize_this_forward to True"
                " and forwarding appropriate data")
        if self.initialize_this_forward:
            self.initialize(x)
            self.initialized = True
            self.initialize_this_forward = False

        loc = self.loc.unsqueeze(0)
        scale = self.scale.unsqueeze(0)
        if len(x.shape) == 4:
            loc = loc.unsqueeze(2).unsqueeze(3)
            scale = scale.unsqueeze(2).unsqueeze(3)
        y = scale * (x + loc)
        return y

    def initialize(self, x):
        with th.no_grad():
            flatten = x.transpose(0,1).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
            )
            std = (
                flatten.std(1)
            )
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-4))
        print("Multiplier initialized to \n", self.scale.squeeze())
