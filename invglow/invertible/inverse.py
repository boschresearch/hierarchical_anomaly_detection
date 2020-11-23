from torch import nn


class Inverse(nn.Module):
    def __init__(self, module):
        super(Inverse, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module.invert(*args, **kwargs)

    def invert(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)
