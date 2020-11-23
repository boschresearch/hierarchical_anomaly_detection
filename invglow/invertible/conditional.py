import torch as th
from torch import nn

class CatChansMerger(nn.Module):
    def __init__(self, cond_preproc=None):
        super(CatChansMerger, self).__init__()
        self.cond_preproc = cond_preproc

    def forward(self, x,cond, fixed=None):
        if self.cond_preproc is not None:
            cond_processed = self.cond_preproc(cond)
        else:
            cond_processed = cond
        return th.cat((x, cond_processed), dim=1)


class ConditionTransformWrapper(nn.Module):
    def __init__(self, module, cond_preproc):
        super().__init__()
        self.module = module
        self.cond_preproc = cond_preproc
        self.accepts_condition=True

    def forward(self, x, condition, fixed=None):
        cond_processed = self.cond_preproc(condition)
        return self.module(x, condition=cond_processed, fixed=fixed)

    def invert(self, y, condition, fixed=None):
        cond_processed = self.cond_preproc(condition)
        return self.module.invert(y, condition=cond_processed, fixed=fixed)


class ApplyAndCat(nn.Module):
    """Apply different modules to different inputs.
    First module will be applied to first input, etc.
    So this module expects to receive a list of inputs
    in the forward."""

    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, xs):
        assert len(xs) == len(self.module_list), (
            f"{len(xs)} xs and {len(self.module_list)} modules")

        ys = [m(x) for m, x in zip(self.module_list, xs)]
        return th.cat(ys, dim=1)
