import torch as th
import numpy as np

# For truncated logic see:
# https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
# torch.fmod(torch.randn(size),2)
def get_gauss_samples(n_samples, mean, std, truncate_to=None):
    if mean.is_cuda:
        orig_samples = th.cuda.FloatTensor(n_samples, len(mean)).normal_(0, 1)
    else:
        orig_samples = th.FloatTensor(n_samples, len(mean)).normal_(0, 1)
    if truncate_to is not None:
        orig_samples = th.fmod(orig_samples, truncate_to)
    orig_samples = th.autograd.Variable(orig_samples)
    samples = (orig_samples * std.unsqueeze(0)) + mean.unsqueeze(0)
    return samples


def get_mixture_gaussian_log_probs(means, log_stds, outs, sum_dims:bool=True,
                                   clamp_max_sigma=None):
    """
    Returns #examples x #mixture components
    """
    demeaned = outs.unsqueeze(1) - means.unsqueeze(0)

    if clamp_max_sigma is not None:
            # unsqueeze over batch dim
            clamp_vals = (th.exp(log_stds.unsqueeze(0)) * clamp_max_sigma)
            # with straight through gradient estimation
            clamped = th.max(th.min(demeaned, clamp_vals), -clamp_vals).detach() + (
                demeaned - demeaned.detach())
    else:
        clamped = demeaned

    unnormed_log_probs = -(clamped ** 2) / (2 * (th.exp(log_stds.unsqueeze(0)) ** 2))
    log_probs = unnormed_log_probs - np.log(np.sqrt(2 * np.pi)) - log_stds.unsqueeze(0)
    if sum_dims:
        log_probs = th.sum(log_probs, dim=2)
    return log_probs


def get_gaussian_log_probs(mean, log_std, outs, sum_dims:bool=True, clamp_max_sigma=None):
    demeaned = outs - mean.unsqueeze(0)
    if clamp_max_sigma is not None:
        # unsqueeze over batch dim
        clamp_vals = (th.exp(log_std) * clamp_max_sigma).unsqueeze(0)
        # with straight through gradient estimation
        clamped = th.max(th.min(demeaned, clamp_vals), -clamp_vals).detach() + (
            demeaned - demeaned.detach())
    else:
        clamped = demeaned

    unnormed_log_probs = -(clamped ** 2) / (2 * (th.exp(log_std) ** 2))
    log_probs = unnormed_log_probs - np.log(np.sqrt(2 * np.pi)) - log_std
    if sum_dims:
        log_probs = th.sum(log_probs, dim=1)
    return log_probs

