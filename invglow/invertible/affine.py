import torch as th
from torch import nn


class AffineCoefs(nn.Module):
    def __init__(self, module, splitter):
        super().__init__()
        self.module = module
        self.splitter = splitter

    def forward(self, x):
        coefs = self.module(x)
        add, raw_scale = self.splitter.split(coefs)
        return (add, raw_scale)

class AdditiveCoefs(nn.Module):
    def __init__(self, module, ):
        super().__init__()
        self.module = module

    def forward(self, x):
        add = self.module(x)
        raw_scale = None
        return (add, raw_scale)


class AffineModifier(nn.Module):
    def __init__(self, sigmoid_or_exp_scale, add_first, eps, ):
        super().__init__()
        self.sigmoid_or_exp_scale = sigmoid_or_exp_scale
        self.add_first = add_first
        self.eps = eps

    def forward(self, x2, coefs):
        add, raw_scale = coefs

        if raw_scale is not None:
            if self.sigmoid_or_exp_scale == 'sigmoid':
                s = th.sigmoid(raw_scale + 2.) + self.eps
            else:
                assert self.sigmoid_or_exp_scale == 'exp'
                s = th.exp(raw_scale) + self.eps
            logdet = th.sum(th.log(s).view(s.shape[0], -1), 1)
        else:
            logdet = 0

        if self.add_first and (add is not None):
            x2 = x2 + add
        if raw_scale is not None:
            x2 = x2 * s
        if (not self.add_first) and (add is not None):
            x2 = x2 + add
        return x2, logdet

    def invert(self, x2, coefs):
        add, raw_scale = coefs
        if raw_scale is not None:
            if self.sigmoid_or_exp_scale == 'sigmoid':
                s = th.sigmoid(raw_scale + 2) + self.eps
            else:
                assert self.sigmoid_or_exp_scale == 'exp'
                s = th.exp(raw_scale) + self.eps
            logdet = th.sum(th.log(s).view(s.shape[0], -1), 1)
        else:
            logdet = 0

        if (not self.add_first) and (add is not None):
            x2 = x2 - add

        if raw_scale is not None:
            x2 = x2 / s

        if (self.add_first) and (add is not None):
            x2 = x2 - add
        return x2, logdet


class AffineBlock(th.nn.Module):
    def __init__(self, FA, FM, single_affine_block,
                 split_merger,
                 sigmoid_or_exp_scale=None,
                 eps=1e-2,
                 condition_merger=None,
                 add_first=None):
        super().__init__()
        if add_first is None:
            print("warning add first is None, setting to False!!")
            add_first = False
        # first G before F, only to have consistent ordering of
        # parameter list compared to other code
        self.FA = FA
        self.FM = FM
        self.split_merger = split_merger
        self.single_affine_block = single_affine_block
        if self.single_affine_block:
            assert self.FM is None
        if (self.FM is not None) or self.single_affine_block:
            assert sigmoid_or_exp_scale is not None
        else:
            assert sigmoid_or_exp_scale is None
        self.sigmoid_or_exp_scale = sigmoid_or_exp_scale
        self.eps = eps
        self.condition_merger = condition_merger
        self.accepts_condition = (condition_merger is not None)
        self.add_first = add_first

    def forward(self, x, condition=None):
        logdet = 0
        x1, x2 = self.split_merger.split(x)
        y1 = x1
        y2 = x2
        if condition is not None:
            assert self.accepts_condition
            y2 = self.condition_merger(y2,condition)

        raw_scale_F = None
        add_F = None
        if self.single_affine_block:
            add_F, raw_scale_F = th.chunk(self.FA(y2), 2, dim=1)
            #h = self.FA(y2)
            #add_F, raw_scale_F = h[:,0::2], h[:,1::2]
        else:
            if self.FA is not None:
                add_F = self.FA(y2)
            if self.FM is not None:
                raw_scale_F = self.FM(y2)

        if raw_scale_F is not None:
            if self.sigmoid_or_exp_scale == 'sigmoid':
                s = th.sigmoid(raw_scale_F + 2.) + self.eps
            else:
                assert self.sigmoid_or_exp_scale == 'exp'
                s = th.exp(raw_scale_F) + self.eps
            logdet = logdet + th.sum(th.log(s).view(s.shape[0], -1), 1)

        if self.add_first and (add_F is not None):
            y1 = y1 + add_F
        if raw_scale_F is not None:
            y1 = y1 * s
        if (not self.add_first) and (add_F is not None):
            y1 = y1 + add_F


        # x2 should be unchanged!!
        y = self.split_merger.merge(y1,x2)
        return y , logdet

    def invert(self, y, condition=None):
        y1, y2 = self.split_merger.split(y)
        x1 = y1
        x2 = y2
        if condition is not None:
            assert self.accepts_condition
            x2 = self.condition_merger(x2,condition)
        logdet = 0

        raw_scale_F = None
        add_F = None
        if self.single_affine_block:
            add_F, raw_scale_F = th.chunk(self.FA(x2), 2, dim=1)
        else:
            if self.FA is not None:
                add_F = self.FA(x2)
            if self.FM is not None:
                raw_scale_F = self.FM(x2)

        if (not self.add_first) and (add_F is not None):
            x1 = x1 - add_F
        if raw_scale_F is not None:
            if self.sigmoid_or_exp_scale == 'sigmoid':
                s = th.sigmoid(raw_scale_F + 2) + self.eps
            else:
                assert self.sigmoid_or_exp_scale == 'exp'
                s = th.exp(raw_scale_F) + self.eps
            logdet = logdet + th.sum(th.log(s).view(s.shape[0], -1), 1)

        if raw_scale_F is not None:
            x1 = x1 / s

        if (self.add_first) and (add_F is not None):
            x1 = x1 - add_F

        # y2 should be unchanged!!
        x = self.split_merger.merge(x1,y2)
        return x, logdet


class AdditiveBlock(AffineBlock):
    def __init__(self, FA, eps=0):
        super(AdditiveBlock, self).__init__(
            FA=FA, FM=None, eps=eps)

class MultiplicativeBlock(AffineBlock):
    def __init__(self, FM, eps=0):
        super(AdditiveBlock, self).__init__(
            FA=None, FM=FM, eps=eps)
