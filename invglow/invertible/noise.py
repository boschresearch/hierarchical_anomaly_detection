import torch as th
from torch import nn
import math


class UniNoise(nn.Module):
    def __init__(self, noise_level=1 / 255.0, center=False):
        super(UniNoise, self).__init__()
        self.noise_level = noise_level
        self.center = center

    def forward(self, x, fixed=None):
        noise = th.rand_like(x)
        if self.center:
            noise = noise - 0.5
        noise = noise * self.noise_level

        return x + noise, 0

    def invert(self, y):
        # can't undo
        return y, 0


class UniformBins(nn.Module):
    def __init__(self, n_bins):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, x):
        x = x + th.zeros_like(x).uniform_(0, 1.0 / self.n_bins)
        log_det = self.compute_log_det(x)
        return x, log_det

    def invert(self, y):
        # can't undo
        log_det = self.compute_log_det(y)
        return y, log_det

    def compute_log_det(self, x):
        b, c, h, w = x.size()
        chw = c * h * w
        log_det = -math.log(self.n_bins) * chw * th.ones(b, device=x.device)
        return log_det


class GaussianNoise(nn.Module):
    def __init__(self, noise_factor=None, means=None, stds=None,):
        super().__init__()
        assert (noise_factor is None) != (
            ((means is None) or (stds is None))
        )
        assert (means is None) ==  (stds is None)
        if means is not None:
            self.register_buffer('means', means)
            self.register_buffer('stds', stds)
            self.noise_factor = None
        else:
            assert noise_factor is not None
            self.noise_factor = noise_factor

    def forward(self, x):
        if self.noise_factor is not None:
            return x + (th.randn_like(x) * self.noise_factor)
        else:
            return x + (th.randn_like(x) * self.stds.unsqueeze(
                0)) + self.means.unsqueeze(0)


class GaussianNoiseGates(nn.Module):
    def __init__(self, n_dims):
        super().__init__()
        self.gates = nn.Parameter(th.ones(n_dims).fill_(2))

    def forward(self, x):
        alphas = th.sigmoid(self.gates)
        rands = th.randn_like(x)
        expanded_alphas = alphas.unsqueeze(0)
        while len(expanded_alphas.shape) < len(x.shape):
            expanded_alphas = expanded_alphas.unsqueeze(-1)
        y = expanded_alphas * x + (1 - expanded_alphas) * rands
        return y, 0

    def invert(self, y):
        return y, 0  # cannot undo noise
