import random
from copy import deepcopy
import logging
import numbers
import torch as th
import numpy as np

log = logging.getLogger(__name__)

def step_and_clear_gradients(optimizer):
    optimizer.step()
    optimizer.zero_grad()


def check_gradients_clear(optimizer):
    for g in optimizer.param_groups:
        for p in g['params']:
            assert p.grad is None or th.all(p.grad.data == 0).item(), (
                "Gradient not none or zero!")


def grads_all_finite(optimizer):
    for g in optimizer.param_groups:
        for p in g['params']:
            if p.grad is None:
                log.warning("Gradient was none on check of finite grads")
            elif not th.all(th.isfinite(p.grad)).item():
                return False
    return True


def clip_to_finite_max(arr, ):
    arr = deepcopy(arr)
    arr[np.isnan(arr)] = np.nanmax(arr)
    arr[~np.isfinite(arr)] = np.max(arr[np.isfinite(arr)])
    return arr


def enforce_2d(outs):
    while len(outs.size()) > 2:
        n_dims = len(outs.size())
        outs = outs.squeeze(2)
        assert len(outs.size()) == n_dims - 1
    return outs


def view_2d(outs):
    return outs.view(outs.size()[0], -1)


def ensure_on_same_device(*variables):
    any_cuda = np.any([v.is_cuda for v in variables])
    if any_cuda:
        variables = [ensure_cuda(v) for v in variables]
    return variables


def ensure_cuda(v):
    if not v.is_cuda:
        v = v.cuda()
    return v



def log_sum_exp(value, dim=None, keepdim=False):
    # https://github.com/pytorch/pytorch/issues/2591#issuecomment-338980717
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = th.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + th.log(th.sum(th.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = th.max(value)
        sum_exp = th.sum(th.exp(value - m))
        return m + th.log(sum_exp)


def set_random_seeds(seed, cuda):
    """
    Set seeds for python random module numpy.random and torch.

    Parameters
    ----------
    seed: int
        Random seed.
    cuda: bool
        Whether to set cuda seed with torch.

    """
    random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def np_to_var(X, requires_grad=False, dtype=None, pin_memory=False,
              **tensor_kwargs):
    """
    Convenience function to transform numpy array to `torch.Tensor`.

    Converts `X` to ndarray using asarray if necessary.

    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor

    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, '__len__'):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = th.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor

def var_to_np(var):
    """Convenience function to transform `torch.Tensor` to numpy
    array.

    Should work both for CPU and GPU."""
    if hasattr(var, 'cpu'):
        return var.cpu().data.numpy()
    else:
        # might happen that you get just a number, in that case nothing to do
        assert isinstance(var, numbers.Number)
        return var


def interpolate_nans_in_df(df):
    df = df.copy()
    for row in df:
        series = np.array(df[row])
        mask = np.isnan(series)
        series[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                 series[~mask])
        df.loc[:, row] = series
    return df


def flatten_2d(a):
    return a.view(len(a), -1)
