import torch as th
import numpy as np

class ChunkChansIn2(object):
    def __init__(self, swap_dims):
        self.swap_dims = swap_dims

    def split(self, x):
        n_chans = x.size()[1]
        assert n_chans % 2 == 0
        if self.swap_dims:
            x1 = x[:, n_chans // 2:]
            x2 = x[:, :n_chans // 2]
        else:
            x1 = x[:, :n_chans // 2]
            x2 = x[:, n_chans // 2:]
        return x1, x2

    def merge(self, y1, x2):
        if self.swap_dims:
            y = th.cat((x2, y1), dim=1)
        else:
            y = th.cat((y1, x2), dim=1)
        return y


class ChansFraction(object):
    def __init__(self, swap_dims, n_unchanged=None, fraction_unchanged=None):
        assert (n_unchanged is None) != (fraction_unchanged is None), (
            "Supply one of n_unchanged or fraction_unchanged")
        self.n_unchanged = n_unchanged
        self.fraction_unchanged = fraction_unchanged
        self.swap_dims = swap_dims

    def split(self, x):
        n_chans = x.size()[1]
        if self.n_unchanged is not None:
            n_unchanged = self.n_unchanged
        else:
            n_unchanged = int(np.round(n_chans * self.fraction_unchanged))
        assert n_unchanged > 0
        assert n_unchanged < n_chans
        if self.swap_dims:
            x1 = x[:, n_unchanged:]
            x2 = x[:, :n_unchanged]

        else:
            x1 = x[:, :-n_unchanged]
            x2 = x[:, -n_unchanged:]
        return x1, x2

    def merge(self, y1, x2):
        if self.swap_dims:
            y = th.cat((x2, y1), dim=1)
        else:
            y = th.cat((y1, x2), dim=1)
        return y


class EverySecondChan(object):
    def split(self, x):
        x1 = x[:,0::2]
        x2 = x[:,1::2]
        return x1, x2

    def merge(self, y1, x2):
        # see also https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/4
        full_shape = (y1.shape[0], y1.shape[1] + x2.shape[1]) + y1.shape[2:]
        y = th.stack((y1,x2), dim=2).view(full_shape)
        return y