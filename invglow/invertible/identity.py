from torch import nn


class Identity(nn.Module):
    def forward(self, x, fixed=None):
        return x, 0

    def invert(self, y, fixed=None):
        return y, 0
