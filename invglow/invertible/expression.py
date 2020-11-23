from torch import nn

class Expression(nn.Module):
    def __init__(self, expression_fn):
        super().__init__()
        self.expression_fn = expression_fn

    def forward(self, x):
        return self.expression_fn(x)