import torch as th
from torch import nn
from invglow.invertible.actnorm import ActNorm
from invglow.invertible.distribution import Unlabeled, NClassIndependentDist
from invglow.invertible.sequential import InvertibleSequential
from invglow.invertible.splitter import SubsampleSplitter
from invglow.invertible.view_as import Flatten2d
from invglow.models.glow import flow_block


def unfold_patches_batch(x, size):
    unfolded = th.nn.functional.unfold(
                x,
                (size, size), stride=(size, size))
    patches = unfolded.reshape(x.shape[0],x.shape[1],size,size,-1)
    patches_batch = patches.permute(0,4,1,2,3).reshape(-1, *patches.shape[1:-1])
    return patches_batch

def fold_to_images_batch(x, n_orig_x, image_size, patch_size, ):
    patches = x.reshape(
        n_orig_x,-1,x.shape[1], patch_size, patch_size).permute(0,2,3,4,1)
    unfolded = patches.reshape(patches.shape[0],-1,patches.shape[-1])
    folded = th.nn.functional.fold(
        unfolded, (image_size,image_size),
        (patch_size,patch_size), stride=(patch_size,patch_size))
    return folded


class WrapForPatches(nn.Module):
    def __init__(self, model, patch_size):
        super().__init__()
        self.model = model
        self.patch_size = patch_size

    def forward(self, x, fixed=None):
        patches = unfold_patches_batch(x, self.patch_size)
        out, lp = self.model(patches)
        image_lp = th.sum(lp.reshape(x.shape[0], -1, *lp.shape[1:]), dim=1)
        return out, image_lp

    def invert(self, z, fixed=None):
        raise ValueError("not implemented")

def create_patch_glow_model(
        hidden_channels,
        K,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        n_chans,
        use_act_norm=True):
    C = n_chans * 4
    H = 4
    W = 4

    splitter = SubsampleSplitter(
        2, via_reshape=True, chunk_chans_first=True, checkerboard=False,
        cat_at_end=True)

    flow_layers = [flow_block(in_channels=C,
                              hidden_channels=hidden_channels,
                              flow_permutation=flow_permutation,
                              flow_coupling=flow_coupling,
                              LU_decomposed=LU_decomposed,
                              cond_channels=0,
                              cond_merger=None,
                              block_type="conv",
                              use_act_norm=use_act_norm) for _ in range(K)]
    flow_this_scale = InvertibleSequential(splitter, *flow_layers)
    flow_this_scale.cuda();
    act_norm = InvertibleSequential(
        Flatten2d(),
        ActNorm((C * H * W),
                scale_fn='exp'))
    dist  = Unlabeled(
                NClassIndependentDist(1, C * H * W, optimize_mean_std=False))
    model = InvertibleSequential(flow_this_scale, act_norm, dist)
    model = WrapForPatches(model, 8)
    return model