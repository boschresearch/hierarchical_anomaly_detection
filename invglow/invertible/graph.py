import torch as th
from torch import nn
import numpy as np
import logging

log = logging.getLogger(__name__)


class AbstractNode(nn.Module):
    def __init__(self, prev, module, notify_prev_nodes=True, **tags):
        super().__init__()
        # Always make into List
        self.change_prev(prev, notify_prev_nodes=notify_prev_nodes)
        self.module = module
        self.cur_out = None
        self.cur_out_log_det = None
        self.cur_in = None
        self.cur_in_log_det = None
        self.tags = tags

    def change_prev(self, prev, notify_prev_nodes):
        if prev is not None:
            if not hasattr(prev, "__len__"):
                prev = [prev]
            prev = nn.ModuleList(prev)
        self.prev = prev
        self.next = []
        if self.prev is not None and notify_prev_nodes:
            for p in self.prev:
                p.register_next(self)

    def register_next(self, next_module):
        self.next.append(next_module)

    def forward(self, x, fixed=None):
        assert self.cur_out is None, "Please remove cur out before forward"
        out, log_det = self._forward(x, fixed=fixed)
        self.remove_cur_in()
        self.remove_cur_out()
        return out, log_det

    def _forward(self, x, fixed=None):
        if self.cur_out is None:
            # Collect incoming results
            if self.prev is not None:
                xs, prev_log_dets = list(zip(*[
                    p._forward(x, fixed=fixed)
                    for p in self.prev]))
                # if has condition node, than here make forward of x already
                if hasattr(self, 'condition_nodes'):
                    for c in self.condition_nodes:
                        _ = c._forward(x, fixed=fixed) # (just ignore output)
            else:
                xs = [x]
                prev_log_dets = [0]

            y, logdet = self._forward_myself(prev_log_dets, *xs,
                                             fixed=fixed)
            self.cur_out = y
            self.cur_out_log_det = logdet
        return self.cur_out, self.cur_out_log_det

    def invert(self, x, fixed=None):
        # determine starting module
        # ps are predecessors
        starting_m = self.find_starting_node()
        inverted = starting_m._invert(x, fixed=fixed)
        self.remove_cur_in()
        self.remove_cur_out()
        return inverted

    def find_starting_node(self):
        cur_ps = [self]
        starting_m = None
        while starting_m is None:
            new_cur_ps = []
            for p in cur_ps:
                if p.prev is None:
                    starting_m = p
                    break
                else:
                    new_cur_ps.extend(p.prev)
            cur_ps = new_cur_ps
        # log.debug("Starting Node" + str(starting_m))
        return starting_m

    def _invert(self, y, fixed=None):
        if self.cur_in is None:
            # Collect incoming results

            if len(self.next) > 0:
                ys = []
                log_dets = []
                for n in self.next:
                    this_y, this_log_det = n._invert(y, fixed=fixed)
                    # Only take those ys belonging to you
                    if len(this_y) > 1 and len(n.prev) > 1:
                        assert len(this_y) == len(n.prev)
                        filtered_y = []
                        for p, a_y in zip(n.prev, this_y):
                            if p == self:
                                filtered_y.append(a_y)
                        this_y = filtered_y
                        if len(this_y) == 1:
                            this_y = this_y[0]
                    ys.append(this_y)
                    log_dets.append(this_log_det)

                # If has condition node, than here make invert of y already

                if hasattr(self, 'condition_nodes'):
                    for c in self.condition_nodes:
                        _ = c._invert(y, fixed=fixed) # (just ignore output)

                # Try to automatically correct ordering in case
                # next nodes are select nodes
                next_class_names = [n.__class__.__name__ for n in self.next]
                if all([n == 'SelectNode' for n in next_class_names]):
                    indices = [n.index for n in self.next]
                    assert np.array_equal(sorted(indices), range(len(ys)))
                    ys = [ys[indices.index(i)] for i in range(len(ys))]

                if (len(ys) == 1) and (self.__class__.__name__ != 'SelectNode' or (
                            not ('no_squeeze' in self.tags and self.tags[
                                'no_squeeze'] == True)
                    )):
                    ys = ys[0]

                next_log_dets = log_dets
            else:
                ys = y
                next_log_dets = [0]
            # log.debug("Now inverting " + str(self.__class__.__name__))
            # if 'name' in self.tags:
            #     log.debug("name: "  + self.tags['name'])
            # if self.module is not None:
            #     log.debug("module: " + str(self.module.__class__.__name__))
            # log.debug("len(self.next) " + str(len(self.next)))
            # log.debug("len(ys) " + str(len(ys)))

            x, log_det = self._invert_myself(next_log_dets, ys, fixed=fixed)
            # Now we save cur out for conditional
            # WARNING: THIS ONLY WORKS IF THE CONDITIONAL NODE ITSELF
            # IS PART OF THE COMPUTATION GRAPH OF THE RESULT _WITHOUT_
            # THE CONDITIONAL NODE PART, SO IT MUST BE USED SOMEWHERE
            # ELSE
            # OTHERWISE THIS CODE NEEDS TO BE ADAPTED SMARTLY
            self.cur_out = ys # possibly necessary for conditional
            self.cur_in = x
            self.cur_in_log_det = log_det
        return self.cur_in, self.cur_in_log_det


    def remove_cur_out(self,):
        if self.prev is not None:
            for p in self.prev:
                p.remove_cur_out()
        if hasattr(self, 'condition_nodes'):
            for c in self.condition_nodes:
                c.remove_cur_out()
                # not sure if necessary
                c.remove_cur_in()
        self.cur_out = None
        self.cur_out_log_det = None

    def remove_cur_in(self,):
        if self.prev is not None:
            for p in self.prev:
                p.remove_cur_in()
        if hasattr(self, 'condition_nodes'):
            for c in self.condition_nodes:
                c.remove_cur_out()
                # not sure if necessary
                c.remove_cur_in()
        self.cur_in = None
        self.cur_in_log_det = None

    def remove_cur_in_out(self):
        self.remove_cur_in()
        self.remove_cur_out()


class Node(AbstractNode):
    def _forward_myself(self, prev_log_dets, *xs, fixed=None):
        y, logdet = self.module(*xs, fixed=fixed)
        prev_sum = sum(prev_log_dets)
        if hasattr(logdet,  'shape') and hasattr(prev_sum, 'shape'):
            if len(logdet.shape) > 1 and len(prev_sum.shape) > 1:
                if logdet.shape[1] == 1 and prev_sum.shape[1] > 1:
                    logdet = logdet.squeeze(1).unsqueeze(1)
                if logdet.shape[1] > 1 and prev_sum.shape[1] == 1:
                    prev_sum = prev_sum.squeeze(1).unsqueeze(1)
            if len(prev_sum.shape) == 1 and len(logdet.shape) == 2:
                prev_sum = prev_sum.unsqueeze(1)
            if len(prev_sum.shape) == 2 and len(logdet.shape) == 1:
                logdet = logdet.unsqueeze(1)

        new_log_det = prev_sum + logdet
        return y, new_log_det

    def _invert_myself(self, next_log_dets, ys, fixed=None):
        # hacky fix
        for i_y in range(len(ys)):
            if isinstance(ys[i_y], tuple):
                ys[i_y] = ys[i_y][0]

        x, log_det = self.module.invert(ys, fixed=fixed)
        return x, sum(next_log_dets) + log_det


class SelectNode(AbstractNode):
    def __init__(self, prev, index, **tags):
        super().__init__(prev, None, notify_prev_nodes=True, **tags)
        self.index = index

    def _forward_myself(self, prev_log_dets, *xs, fixed=None):
        # don't understand reason for next two lines
        assert len(xs) == 1
        xs = xs[0]
        n_parts = len(xs)
        assert n_parts > self.index
        return xs[self.index], sum(prev_log_dets) / n_parts

    def _invert_myself(self, next_log_dets, ys, fixed=None):
        return ys, sum(next_log_dets)


class CatChansNode(AbstractNode):
    def __init__(self, prev, notify_prev_nodes=True, **tags):
        self.n_chans = None
        super(CatChansNode, self).__init__(prev, None,
                                           notify_prev_nodes=notify_prev_nodes,
                                         **tags)

    def _forward_myself(self, prev_log_dets, *xs, fixed=None):
        n_chans = tuple([a_x.size()[1] for a_x in xs])
        if self.n_chans is None:
            self.n_chans = n_chans
        else:
            assert n_chans == self.n_chans
        return th.cat(xs, dim=1), sum(prev_log_dets)

    def _invert_myself(self, next_log_dets, ys, fixed=None):
        if self.n_chans is None:
            n_parts = len(self.prev)
            xs = th.chunk(ys, chunks=n_parts, dim=1, )
            self.n_chans = tuple([a_x.size()[1] for a_x in xs])
        else:
            xs = []
            bounds = np.insert(np.cumsum(self.n_chans), 0, 0)
            for i_b in range(len(bounds) - 1):
                xs.append(ys[:, bounds[i_b]:bounds[i_b + 1]])
        return xs, sum(next_log_dets) / len(xs)


class ConditionalNode(AbstractNode):
    def __init__(self, prev, module, condition_nodes, **tags):
        super().__init__(prev, module, notify_prev_nodes=True,
                                              **tags)
        assert any([hasattr(m, 'accepts_condition') and m.accepts_condition
                    for m in module.modules()])
        if not hasattr(condition_nodes, '__len__'):
            condition_nodes = [condition_nodes]
        self.condition_nodes = condition_nodes

    def get_condition(self):
        for c in self.condition_nodes:
            assert c.cur_out is not None
        condition = [c.cur_out for c in self.condition_nodes]
        if len(condition) == 1:
            condition = condition[0]
        return condition

    def _forward_myself(self, prev_log_dets, *xs, fixed=None):
        condition = self.get_condition()
        y, logdet = self.module(
            *xs, condition=condition, fixed=fixed)
        return y, sum(prev_log_dets) + logdet

    def _invert_myself(self, next_log_dets, ys, fixed=None):
        condition = self.get_condition()
        x, log_det = self.module.invert(
            ys, condition=condition, fixed=fixed)
        return x, sum(next_log_dets) + log_det


class IntermediateResultsNode(AbstractNode):
    def __init__(self, prev, **tags):
        self.n_chans = None
        super().__init__(prev, None, notify_prev_nodes=False, **tags)

    def _forward_myself(self, prev_log_dets, *xs, fixed=None):
        return xs, prev_log_dets

    def _invert_myself(self, next_log_dets, ys, fixed=None):
        return ys, next_log_dets


class CatAsListNode(AbstractNode):
    def __init__(self, prev, notify_prev_nodes=True, **tags):
        super().__init__(prev, None, notify_prev_nodes=notify_prev_nodes, **tags)

    def _forward_myself(self, prev_log_dets, *xs, fixed=None):
        max_len_shape = max([len(l.shape) for l in prev_log_dets])
        new_prev_log_dets = []
        for p in prev_log_dets:
            if hasattr(p, 'shape') and len(p.shape) < max_len_shape:
                p = p.unsqueeze(1)
            new_prev_log_dets.append(p)
        return xs, sum(new_prev_log_dets)

    def _invert_myself(self, next_log_dets, ys, fixed=None):
        # log.debug("Inverting cat as list node NOW,,,,,")
        # log.debug("in cat as list len(ys)" + str(len(ys)))
        return ys, sum(next_log_dets) / len(ys)


class MergeLogDetsNode(AbstractNode):
    def __init__(self, prev, notify_prev_nodes=True, **tags):
        super().__init__(prev, None, notify_prev_nodes=notify_prev_nodes,
                         **tags)

    def _forward_myself(self, prev_log_dets, *xs, fixed=None):
        if fixed['y'] is None:
            if not hasattr(prev_log_dets, 'shape'):
                prev_log_dets = prev_log_dets[0]
            n_components = prev_log_dets.shape[1]
            logdets = th.logsumexp(prev_log_dets, dim=1) - np.log(n_components)


        return xs, logdets

    def _invert_myself(self, next_log_dets, ys, fixed=None):
        raise NotImplementedError("Check if you can just pass it through")
        return ys, next_log_dets


def get_all_nodes(final_node):
    cur_ps = [final_node]
    all_nodes = []
    while len(cur_ps) > 0:
        new_cur_ps = []
        for p in cur_ps:
            if p.prev is not None:
                new_cur_ps.extend(p.prev)
            if p not in all_nodes:
                all_nodes.append(p)
        cur_ps = new_cur_ps
    return all_nodes[::-1]


def get_nodes_by_tags(full_model, **tags):
    nodes = []
    for n in get_all_nodes(full_model):
        put_inside = True
        for tag in tags:
            if (tag not in n.tags) or n.tags[tag] != tags[tag]:
                put_inside = False
        if put_inside:
            nodes.append(n)
    return nodes


def get_nodes_by_names(full_model, *names):
    name_to_node = dict()
    for n in get_all_nodes(full_model):
        if 'name' in n.tags and n.tags['name'] in names:
            name_to_node[n.tags['name']] = n
    nodes = [name_to_node[name] for name in names]
    return nodes
