import torch as th
import numpy as np


class ViewAs(th.nn.Module):
    def __init__(self, dims_before, dims_after):
        super().__init__()
        self.dims_before = dims_before
        self.dims_after = dims_after

    def forward(self, x, fixed=None):
        for i_dim in range(len(x.size())):
            expected = self.dims_before[i_dim]
            if expected != -1:
                assert x.size()[i_dim] == expected, (
                    "Expected size {:s}, Actual: {:s}".format(
                        str(self.dims_before), str(x.size()))
                )
        return x.view(self.dims_after), 0

    def invert(self, features, fixed=None):
        for i_dim in range(len(features.size())):
            expected = self.dims_after[i_dim]
            if expected != -1:
                assert features.size()[i_dim] == expected, (
                    "Expected size {:s}, Actual: {:s}".format(
                        str(self.dims_after), str(features.size()))
                )
        features = features.view(self.dims_before)
        return features, 0

    def __repr__(self):
        return "ViewAs({:s}, {:s})".format(
            str(self.dims_before), str(self.dims_after))


class Flatten2d(th.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.dims_before = None

    def forward(self, x, fixed=None):
        self.dims_before = x.size()
        y = x.view(x.size()[0], -1)
        return y, 0

    def invert(self, features, fixed=None):
        assert self.dims_before is not None, (
            "Please call forward first")
        features = features.view(-1,
                                 *self.dims_before[1:])
        return features, 0

    def __repr__(self):
        return "Flatten2d({:s}".format(
            str(self.dims_before))


class Flatten2dAndCat(th.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.dims_before = None

    def forward(self, x, fixed=None):
        self.dims_before = [a_x.shape for a_x in x]
        y = th.cat([a_x.contiguous().view(a_x.size()[0], -1) for a_x in x], dim=1)
        return y, 0

    def invert(self, features, fixed=None):
        assert self.dims_before is not None, (
            "Please call forward first")
        xs = []
        i_start = 0
        for shape in self.dims_before:
            n_len = int(np.prod(shape[1:]))
            part_f = features[:, i_start:i_start+n_len]
            xs.append(part_f.view(-1,*shape[1:]))
            i_start += n_len
        return xs, 0

    def __repr__(self):
        return "Flatten2dAndCat({:s})".format(
            str(self.dims_before))
