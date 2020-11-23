import torch as th
import torch.nn.functional as F

def nll_class_loss(base_nll, fine_nll, target_val,
                   temperature, weight, reduction):
    assert target_val in [0, 1]
    ldiff = -(fine_nll - base_nll.detach())
    return nll_diff_loss(ldiff, target_val=target_val,
                         temperature=temperature,
                         weight=weight,
                         reduction=reduction)


def nll_diff_loss(lp_diff, target_val, temperature, weight,
                  reduction):
    class_loss = F.binary_cross_entropy_with_logits(
        lp_diff / temperature, th.zeros_like(lp_diff) + target_val,
        reduction=reduction)
    class_loss = weight * class_loss
    return class_loss