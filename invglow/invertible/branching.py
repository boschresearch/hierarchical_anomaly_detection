import numpy as np
import torch as th
from torch import nn


class ChunkChans(nn.Module):
    def __init__(self, n_parts):
        super(ChunkChans, self).__init__()
        self.n_parts = n_parts

    def forward(self, x, fixed=None):
        xs = th.chunk(x, chunks=self.n_parts, dim=1, )
        # for debug
        self.my_x_sizes = [x.size() for x in xs]
        return xs, 0

    def invert(self, y, fixed=None):
        y = th.cat(y, dim=1)
        return y, 0


class SwitchX1X2(nn.Module):
    def forward(self, x):
        x1, x2 = th.chunk(x, 2, dim=1)
        return th.cat([x2, x1], dim=1)

    def invert(self, y):
        return self.forward(y)


class ChunkByIndex(nn.Module):
    def __init__(self, index):
        super(ChunkByIndex, self).__init__()
        self.index = index

    def forward(self, x, fixed=None):
        xs = [x[:, :self.index], x[:,self.index:]]
        return xs, 0

    def invert(self, y, fixed=None):
        y = th.cat(y, dim=1)
        return y, 0

class ChunkByIndices(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = tuple(indices)

    def forward(self, x, fixed=None):
        indices = (0,) + self.indices + (x.shape[1],)
        xs = [x[:, start:stop]
              for start, stop in zip(indices[:-1], indices[1:])]
        return xs, 0

    def invert(self, y, fixed=None):
        y = th.cat(y, dim=1)
        return y, 0


class CatChans(nn.Module):
    def __init__(self,):
        super().__init__()
        self.n_chans = None

    def forward(self, xs, fixed=None):
        n_chans = tuple([a_x.size()[1] for a_x in xs])
        if self.n_chans is None:
            self.n_chans = n_chans
        else:
            assert n_chans == self.n_chans
        return th.cat(xs, dim=1), 0

    def invert(self, ys, fixed=None):
        assert self.n_chans is not None, "please do forward first"
        if ys is not None:
            xs = []
            bounds = np.insert(np.cumsum(self.n_chans), 0, 0)
            for i_b in range(len(bounds) - 1):
                xs.append(ys[:, bounds[i_b]:bounds[i_b + 1]])
        else:
            xs = [None] * len(self.n_chans)
        return xs, 0
