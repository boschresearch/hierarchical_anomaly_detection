# Partly from https://github.com/rosinality/glow-pytorch/

import numpy as np

from scipy import linalg as la
import torch.nn.functional as F
import torch as th
from torch import nn


class InvPermute(nn.Module):
    def __init__(self, in_channel, fixed, use_lu):
        super().__init__()
        self.use_lu = use_lu
        self.fixed = fixed
        if not use_lu:
            weight = th.randn(in_channel, in_channel)
            q, _ = th.qr(weight)
            weight = q
            if fixed:
                self.register_buffer('weight', weight.data)
                self.register_buffer('weight_inverse', weight.data.inverse())
                self.register_buffer('fixed_log_det',
                                     th.slogdet(self.weight.double())[1].float())
            else:
                self.weight = nn.Parameter(weight)
        if use_lu:
            assert not fixed
            #weight = np.random.randn(in_channel, in_channel)
            weight = th.randn(in_channel, in_channel)
            #q, _ = la.qr(weight)
            q, _ = th.qr(weight)

            # w_p, w_l, w_u = la.lu(q.astype(np.float32))
            w_p, w_l, w_u = th.lu_unpack(*th.lu(q))

            #w_s = np.diag(w_u)
            w_s = th.diag(w_u)
            #w_u = np.triu(w_u, 1)
            w_u = th.triu(w_u, 1)
            #u_mask = np.triu(np.ones_like(w_u), 1)
            u_mask = th.triu(th.ones_like(w_u), 1)
            #l_mask = u_mask.T
            l_mask = u_mask.t()

            #w_p = th.from_numpy(w_p)
            #w_l = th.from_numpy(w_l)
            #w_s = th.from_numpy(w_s)
            #w_u = th.from_numpy(w_u)

            self.register_buffer('w_p', w_p)
            self.register_buffer('u_mask', u_mask)
            self.register_buffer('l_mask', l_mask)
            self.register_buffer('s_sign', th.sign(w_s))
            self.register_buffer('l_eye', th.eye(l_mask.shape[0]))
            self.w_l = nn.Parameter(w_l)
            self.w_s = nn.Parameter(th.log(th.abs(w_s)))
            self.w_u = nn.Parameter(w_u)

    def reset_to_identity(self):
        def eye_like(w):
            return th.eye(
                len(w), device=w.device,
                dtype=w.dtype)
        if self.use_lu:
            self.w_p.data.copy_(eye_like(self.w_p))
            self.s_sign.data.copy_(th.ones_like((self.s_sign)))
            self.w_l.data.copy_(eye_like(self.w_l))
            self.w_s.data.copy_(th.ones_like((self.w_s)))
            self.w_u.data.zero_()

        else:
            self.weight.data.copy_(eye_like(self.weight))
            if self.fixed:
                self.weight_inverse.data.copy_(eye_like(self.weight))
                self.fixed_log_det.copy_(th.zeros_like(self.weight[0,0]))

    def forward(self, x, fixed=None):
        weight = self.calc_weight()
        if len(x.shape) == 2:
            y = F.linear(x, weight)
        else:
            assert len(x.shape) == 4
            y = F.conv2d(x, weight.unsqueeze(2).unsqueeze(3))

        logdet = self.compute_log_det(x.shape)
        return y, logdet

    def calc_weight(self):
        if self.use_lu:
            weight = (
                    self.w_p
                    @ (self.w_l * self.l_mask + self.l_eye)
                    @ ((self.w_u * self.u_mask) + th.diag(
                self.s_sign * th.exp(self.w_s)))
            )
        else:
            weight = self.weight
        return weight

    def invert(self, y, fixed=None):
        if self.fixed:
            weight_inverse = self.weight_inverse
        else:
            weight = self.calc_weight()
            weight_inverse = weight.inverse()
        if len(y.shape) == 2:
            x = F.linear(y, weight_inverse)
        else:
            assert len(y.shape) == 4
            x = F.conv2d(y, weight_inverse.unsqueeze(2).unsqueeze(3))
        logdet = self.compute_log_det(x.shape)
        return x, logdet

    def compute_log_det(self, x_shape):
        logdet = self.compute_log_det_per_px()
        if len(x_shape) == 4:
            _, _, height, width = x_shape
            logdet = logdet * height * width
        else:
            assert len(x_shape) == 2
        return logdet

    def compute_log_det_per_px(self):
        if self.fixed:
            logdet = self.fixed_log_det
        else:
            if self.use_lu:
                logdet = th.sum(self.w_s)
            else:
                logdet = th.slogdet(self.weight.double())[1].float()
        return logdet

class Shuffle(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        indices = th.randperm(in_channel)
        invert_inds = th.sort(indices)[1]
        self.register_buffer('indices', indices)
        self.register_buffer('invert_inds', invert_inds)

    def forward(self, x, fixed=None):
        assert x.shape[1] == len(self.indices)
        y = x[:, self.indices]
        return y,0

    def invert(self, y, fixed=None):
        assert y.shape[1] == len(self.indices)
        x = y[:, self.invert_inds]
        return x,0