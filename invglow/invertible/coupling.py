from torch import nn


class CouplingLayer(nn.Module):
    def __init__(self, split_merger, coef_extractor,
                 modifier,
                 condition_merger=None,
                 ):
        super().__init__()
        self.split_merger = split_merger
        self.coef_extractor = coef_extractor
        self.modifier = modifier
        self.condition_merger = condition_merger
        self.accepts_condition = (condition_merger is not None)

    def forward(self, x, condition=None, fixed=None):
        x1, x2 = self.split_merger.split(x)
        y1 = x1
        y2 = x2
        if condition is not None:
            assert self.accepts_condition
            y2 = self.condition_merger(y2, condition)
        coefs = self.coef_extractor(y2)
        y1, log_det = self.modifier(y1, coefs)
        y = self.split_merger.merge(y1, x2)  # x2 should be unchanged
        return y, log_det

    def invert(self, y, condition=None, fixed=None):
        y1, y2 = self.split_merger.split(y)
        x1 = y1
        x2 = y2
        if condition is not None:
            assert self.accepts_condition
            x2 = self.condition_merger(x2, condition)
        coefs = self.coef_extractor(x2)
        x1, log_det = self.modifier.invert(x1, coefs)
        x = self.split_merger.merge(x1, y2)  # y2 should be unchanged
        return x, log_det