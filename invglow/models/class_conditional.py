import logging
from invglow.invertible.actnorm import ActNorm
from invglow.invertible.branching import ChunkByIndices
from invglow.invertible.distribution import Unlabeled, NClassIndependentDist
from invglow.invertible.graph import CatAsListNode
from invglow.invertible.graph import Node, SelectNode
from invglow.invertible.sequential import InvertibleSequential
from invglow.invertible.view_as import Flatten2d

log = logging.getLogger('__name__')

def latent_model(n_chans):
    n_dims_per_scale = [n_chans*2*16*16, n_chans*4*8*8, n_chans*16*4*4]
    rechunker = ChunkByIndices((n_dims_per_scale[0], sum(n_dims_per_scale[:2])))
    nd_in_split_again = Node(None, rechunker)
    dist_nodes = []
    for i_scale in range(3):
        n_dims_scale = n_dims_per_scale[i_scale]
        nd_in_class = SelectNode(nd_in_split_again, i_scale)
        act_class = InvertibleSequential(
                Flatten2d(),
                ActNorm(n_dims_scale, scale_fn='exp'))
        dist_class = Unlabeled(
          NClassIndependentDist(1, n_dims_scale))
        nd_act_class = Node(nd_in_class,act_class)
        nd_dist_class = Node(nd_act_class, dist_class)
        dist_nodes.append(nd_dist_class)
    top_model = CatAsListNode(dist_nodes)
    return top_model



def convert_class_model_for_multi_scale_nll(model):
    log.warning("Please be aware that these results are not mathematically correct")
    pre_class_nodes_per_scale = model.sequential[0].prev
    for p in pre_class_nodes_per_scale:
        p.next = []

    class_model = model.sequential[1].module
    per_scale_act_dist_mods = [[], [], []]
    for single_model in class_model.module_list:
        per_scale_nodes = single_model.prev[0].prev
        for i_scale, per_scale_node in enumerate(per_scale_nodes):
            act_mod = per_scale_node.prev[0].module
            dist_mod = per_scale_node.module
            per_scale_act_dist_mods[i_scale].append(
                InvertibleSequential(act_mod, dist_mod))

    act_dist_nodes_per_scale = [
        MergeLogDetsNode(Node(
            pre_class_nodes_per_scale[i_scale],
            InvertibleClassConditional(
                per_scale_act_dist_mods[i_scale],
                 i_classes=list(range(len(per_scale_act_dist_mods[i_scale])))),
            name=f'm0-dist-{i_scale}'))
        for i_scale in range(3)]
    model = CatAsListNode(act_dist_nodes_per_scale)
    return model