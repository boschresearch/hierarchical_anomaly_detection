from torch import nn
import torch as th

class InvertibleClassConditional(nn.Module):
    def __init__(self, modules, i_classes):
        super().__init__()
        self.module_list = nn.ModuleList(modules)
        self.i_classes = i_classes

    def forward(self, x, fixed):
        return self.compute(x, fixed=fixed, mode='forward')

    def invert(self, y, fixed):
        return self.compute(y, fixed=fixed, mode='invert')

    def compute(self, x, fixed, mode):
        y = fixed['y']
        if y is None:  # just compute all mixture components for all examples
            if hasattr(x, 'shape'): # if it was list already
                # then no need to split up again, we already duplicated it...
                xs = [x] * len(self.module_list)
            else:
                xs = x
        else:
            masks = [y[:, i_class] == 1 for i_class in self.i_classes]
            xs = [x[m] for m in masks]

        outs = []
        log_dets = []
        assert len(xs) == len(self.module_list)
        for a_x, module in zip(xs, self.module_list):
            if len(a_x) > 0:
                if mode == 'forward':
                    this_out, this_log_det = module(a_x,
                                                    fixed=fixed)
                    o_shape = this_out.shape  # used below
                else:
                    assert mode == 'invert'
                    this_out, this_log_det = module.invert(a_x,
                                                           fixed=fixed)
            else:
                this_out, this_log_det = None, None
            outs.append(this_out)
            log_dets.append(this_log_det)

        if y is None:
            return outs, th.stack(log_dets, dim=-1)
        else:
            assert len(outs) == len(masks) == len(log_dets), (
                f"n_outs: {len(outs)}, n_masks: {len(masks)}, n_dets: {len(log_dets)}")
            outs_full = th.zeros(len(x), *o_shape[1:], dtype=x.dtype,
                                 device=x.device)
            log_dets_full = th.zeros(len(x), dtype=x.dtype, device=x.device)
            for out, log_det, mask in zip(outs, log_dets, masks):
                if out is not None:
                    #outs_full[mask] = outs_full[mask] + out
                    #log_dets_full[mask] = log_dets_full[mask] + log_det
                    log_dets_full = log_dets_full.masked_scatter(mask, log_det)
                    while len(mask.shape) < len(outs_full.shape):
                        mask = mask.unsqueeze(-1).repeat(
                            (1,) * len(mask.shape) + (
                            outs_full.shape[len(mask.shape)],))
                    outs_full = outs_full.masked_scatter(mask, out)

            # counts = th.zeros(len(self.i_classes), dtype=th.int64)
            # y_label = th.argmax(y, dim=1)
            # all_outs = []
            # all_log_dets = []
            # for i in range(len(x)):
            #     i_class = y_label[i]
            #     i_in_class = counts[i_class]
            #     all_outs.append(outs[i_class][i_in_class])
            #     all_log_dets.append(log_dets[i_class][i_in_class])
            #     counts[i_class] += 1
            #
            # outs_full = th.stack(all_outs, axis=0)
            # log_dets_full = th.stack(all_log_dets, axis=0)
            #

            # outs_full = th.cat(outs, axis=0)
            # log_dets_full = th.cat(log_dets, axis=0)
            return outs_full, log_dets_full
