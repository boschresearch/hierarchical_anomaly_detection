import logging
from collections import OrderedDict

import numpy as np
import torch as th
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision import transforms
from tqdm import tqdm

from invglow.datasets import load_tiny_imagenet, load_train_test, LSUN, load_celeb_a
from invglow.invertible.graph import IntermediateResultsNode
from invglow.invertible.graph import get_nodes_by_names
from invglow.models.class_conditional import \
    convert_class_model_for_multi_scale_nll
from invglow.util import var_to_np, set_random_seeds
from invglow.datasets import PreprocessedLoader
from invglow.invertible.expression import  Expression
from invglow import folder_locations

log = logging.getLogger(__name__)


def get_nlls(loader, wanted_nodes, node_names):
    rs = []
    with th.no_grad():
        for x, y in tqdm(loader):
            outs = IntermediateResultsNode(wanted_nodes)(
                x.cuda(),fixed=dict(y=None))
            lps = outs[1]
            fixed_lps = []
            for lp in lps:
                if len(lp.shape) > 1:
                    assert len(lp.shape) == 2
                    n_components = lp.shape[1]
                    lp = th.logsumexp(lp, dim=1) - np.log(n_components)
                fixed_lps.append(lp)
            node_to_lps = dict(zip(node_names, fixed_lps))
            lp0 = node_to_lps['m0-flow-0'] / 2 + node_to_lps['m0-dist-0']
            lp1 = node_to_lps['m0-flow-1'] / 2 + node_to_lps['m0-dist-1'] - \
                  node_to_lps['m0-flow-0'] / 2
            lp2 = node_to_lps['m0-dist-2'] - node_to_lps['m0-flow-1'] / 2
            lpz0 = node_to_lps['m0-dist-0'] - node_to_lps['m0-act-0']
            lpz1 = node_to_lps['m0-dist-1'] - node_to_lps['m0-act-1']
            lpz2 = node_to_lps['m0-dist-2'] - node_to_lps['m0-act-2']
            lp0 = lp0.cpu().numpy()
            lp1 = lp1.cpu().numpy()
            lp2 = lp2.cpu().numpy()
            lpz0 = lpz0.cpu().numpy()
            lpz1 = lpz1.cpu().numpy()
            lpz2 = lpz2.cpu().numpy()
            lprob = lp0 + lp1 + lp2
            lprobz = lpz0+lpz1+lpz2
            bpd = np.log2(256) - ((lprob / np.log(2)) / np.prod(x.shape[1:]))
            rs.append(dict(lp0=lp0, lp1=lp1, lp2=lp2, lprob=lprob, bpd=bpd,
                          lpz0=lpz0, lpz1=lpz1, lpz2=lpz2, lprobz=lprobz))
    full_r = {}
    for key in rs[0].keys():
        full_r[key] = np.concatenate([r[key] for r in rs])
    return full_r


def get_nlls_only_final(loader, model):
    rs = []
    with th.no_grad():
        for x, y in tqdm(loader):
            _, lp = model(x.cuda(), fixed=dict(y=None))
            lprob = var_to_np(lp)
            bpd = np.log2(256) - ((lprob / np.log(2)) / np.prod(x.shape[1:]))
            rs.append(dict(lprob=lprob, bpd=bpd,))
    full_r = {}
    for key in rs[0].keys():
        full_r[key] = np.concatenate([r[key] for r in rs])
    return full_r



def get_rgb_loaders(first_n=None):
    train_cifar10, test_cifar10 = load_train_test('cifar10', shuffle_train=False,
                                              drop_last_train=False,
                                              batch_size=512,
                                              eval_batch_size=512,
                                              n_workers=6,
                                              first_n=first_n,
                                              augment=False,
                                              exclude_cifar_from_tiny=None,)

    train_svhn, test_svhn = load_train_test('svhn', shuffle_train=False,
                                            drop_last_train=False,
                                            batch_size=512, eval_batch_size=512,
                                            n_workers=6,
                                            first_n=first_n,
                                            augment=False,
                                            exclude_cifar_from_tiny=None,)

    categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                  'conference_room', 'dining_room', 'kitchen',
                  'living_room', 'restaurant', 'tower']

    def final_preproc_lsun(x):
        # make same as cifar tiny etc.
        return (x * 255 / 256) - 0.5

    lsun_set = LSUN(folder_locations.lsun_data,
                    classes=[c + '_val' for c in categories],
                    transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.CenterCrop(32),
                        transforms.ToTensor(),
                        final_preproc_lsun]))

    test_lsun = th.utils.data.DataLoader(lsun_set, batch_size=512,
                                         num_workers=0)
    train_cifar100, test_cifar100 = load_train_test('cifar100',
                                                    shuffle_train=False,
                                                    drop_last_train=False,
                                                    batch_size=512,
                                                    eval_batch_size=512,
                                                    n_workers=6,
                                                    first_n=first_n,
                                                    augment=False,
                                                    exclude_cifar_from_tiny=None,)
    #train_tiny, test_tiny = load_train_test(
    #    'tiny',
    #    shuffle_train=False,
    #    drop_last_train=False,
    #    batch_size=512, eval_batch_size=512,
    #    n_workers=6,
    #    first_n=120000 if first_n is None else first_n,
    #    augment=False,
    #    exclude_cifar_from_tiny=False,
    #    shuffle_tiny_chunks=False)
    train_celeba, test_celeba = load_celeb_a(shuffle_train=False,
                                             drop_last_train=False,
                                             batch_size=512,
                                             eval_batch_size=512,
                                             n_workers=6,
                                             first_n=first_n, )
    train_tiny_imagenet, test_tiny_imagenet = load_tiny_imagenet(
        shuffle_train=False, drop_last_train=False,
        batch_size=512, eval_batch_size=512,
        n_workers=6,
        first_n=first_n, )



    loaders = dict(
        train_cifar10=train_cifar10,
        test_cifar10=test_cifar10,
        train_svhn=train_svhn,
        test_svhn=test_svhn,
        #train_tiny=train_tiny,
        #test_tiny=test_tiny,
        test_lsun=test_lsun,
        train_cifar100=train_cifar100,
        test_cifar100=test_cifar100,
        train_celeba=train_celeba,
        test_celeba=test_celeba,
        train_tiny_imagenet=train_tiny_imagenet,
        test_tiny_imagenet=test_tiny_imagenet,
    )
    return loaders


def get_grey_loaders(first_n):
    train_mnist, test_mnist = load_train_test('mnist',
        shuffle_train=False,
        drop_last_train=False,
        batch_size=512,
        eval_batch_size=512,
        n_workers=6,
        first_n=first_n,
        augment=False,
        exclude_cifar_from_tiny=None,)
    train_fashion_mnist, test_fashion_mnist = load_train_test('fashion-mnist',
        shuffle_train=False,
        drop_last_train=False,
        batch_size=512,
        eval_batch_size=512,
        n_workers=6,
        first_n=first_n,
        augment=False,
        exclude_cifar_from_tiny=None,)

    #train_tiny, test_tiny = load_train_test(
    #    'tiny',
    #    shuffle_train=False,
    #    drop_last_train=False,
    #    batch_size=512, eval_batch_size=512,
    #    n_workers=6,
    #    first_n=120000 if first_n is None else first_n,
    #    augment=False,
    #    exclude_cifar_from_tiny=False,
    #    shuffle_tiny_chunks=False,
    #    tiny_grey=True)


    loaders = dict(
        train_mnist=train_mnist,
        test_mnist=test_mnist,
        #train_tiny=train_tiny,
        #test_tiny=test_tiny,
        train_fashion_mnist=train_fashion_mnist,
        test_fashion_mnist=test_fashion_mnist,
    )
    return loaders


def set_non_finite_to(arr, val):
    arr = arr.copy()
    arr[~np.isfinite(arr)] = val
    return arr


def evaluate_without_noise(fine_model, base_model, on_top_class,
                           first_n, noise_factor,
                           in_dist_name, rgb_or_grey,
                           only_full_nll, ):
    fine_results = _evaluate_without_noise(
        fine_model, on_top_class=on_top_class,
        first_n=first_n, noise_factor=noise_factor,
        rgb_or_grey=rgb_or_grey,
        only_full_nll=only_full_nll, )
    base_results = _evaluate_without_noise(
        base_model, on_top_class=False,
        first_n=first_n, noise_factor=noise_factor,
        rgb_or_grey=rgb_or_grey,
        only_full_nll=only_full_nll, )
    in_dist_diff = set_non_finite_to(
        fine_results['test_' + in_dist_name]['lprob'],
        -3000000) - set_non_finite_to(
        base_results['test_' + in_dist_name]['lprob'],
        -3000000)
    if not only_full_nll:
        in_dist_4x4 = set_non_finite_to(
            fine_results['test_' + in_dist_name]['lp2'], -3000000)

    if rgb_or_grey == 'rgb':
        ood_sets = ['cifar10', 'cifar100', 'svhn', 'lsun', 'celeba', 'tiny_imagenet']
    else:
        ood_sets = ['fashion_mnist', 'mnist']
    ood_sets = [s for s in ood_sets if s != in_dist_name]
    for ood_set in ood_sets:
        folds = ('test',)
        if ood_set == 'celeba':
            folds = ('train', 'test',)
        ood_diff = np.concatenate([set_non_finite_to(
            fine_results[f'{fold}_{ood_set}']['lprob'],
            -3000000) - set_non_finite_to(
            base_results[f'{fold}_{ood_set}']['lprob'],
            -3000000) for fold in folds])
        auc = compute_auc_for_scores(ood_diff, in_dist_diff)
        print(f"{ood_set}: {auc:.1%} ratio AUC")
        if not only_full_nll:
            ood_4x4 = np.concatenate([set_non_finite_to(
                fine_results[f'{fold}_{ood_set}']['lp2'], -3000000)
                for fold in folds])
            auc_4x4 = compute_auc_for_scores(ood_4x4, in_dist_4x4)
            print(f"{ood_set}: {auc_4x4:.1%} 4x4 AUC")


def _evaluate_without_noise(model, on_top_class,
                            first_n, noise_factor, rgb_or_grey,
                            only_full_nll, ):
    assert rgb_or_grey in ["rgb", "grey"]
    if rgb_or_grey == 'rgb':
        loaders = get_rgb_loaders(first_n=first_n)
    else:
        assert rgb_or_grey == 'grey'
        loaders = get_grey_loaders(first_n=first_n)

    if on_top_class:
        model = convert_class_model_for_multi_scale_nll(model)

    if not only_full_nll:
        node_names = ('m0-flow-0', 'm0-act-0', 'm0-dist-0',
                      'm0-flow-1', 'm0-act-1', 'm0-dist-1',
                      'm0-flow-2', 'm0-act-2', 'm0-dist-2')
        try:
            wanted_nodes = get_nodes_by_names(model, *node_names)
        except:
            wanted_nodes = get_nodes_by_names(model.module, *node_names)

    loaders_to_results = {}
    for set_name, loader in loaders.items():
        set_random_seeds(20191120, True)
        # add half of noise interval
        loader = PreprocessedLoader(loader,
                                    Expression(lambda x: x + noise_factor / 2),
                                    to_cuda=True)
        if not only_full_nll:
            print(set_name)
            result = get_nlls(loader, wanted_nodes, node_names)
        else:
            result = get_nlls_only_final(loader, model)
        loaders_to_results[set_name] = result
    return loaders_to_results


def compute_func_for_sets(name_to_train_test_loaders, model_dist, func):
    results_per_set = OrderedDict()
    for setname, train_loader, test_loader in name_to_train_test_loaders:
        for name, loader in (('train', train_loader), ('test', test_loader)):
            if loader is None: continue
            results = func(loader, model_dist)
            results_per_set[name + '_' + setname] = results
    return results_per_set


def get_log_dets_probs_for_set(loader, model_dist):
    with th.no_grad():
        n_examples = sum([len(x) for x,y in loader])
        n_components = len(model_dist.dist.class_means)
        all_log_probs = np.ones((n_examples,n_components)) * np.nan
        all_log_dets = np.ones((n_examples,)) * np.nan
        i_example = 0
        for x,y in loader:
            x = x.cuda()
            out, log_det = model_dist.model(x)
            log_probs_per_class = model_dist.dist.log_probs_per_class(out)
            all_log_probs[i_example:i_example + len(x)] = var_to_np(log_probs_per_class)
            all_log_dets[i_example:i_example + len(x)] = var_to_np(log_det)
            i_example += len(x)
        assert not np.any(np.isnan(all_log_probs))
        assert not np.any(np.isnan(all_log_dets))
    return all_log_probs, all_log_dets


def get_in_diffs_per_set(loader, model_dist):
    n_examples = sum([len(x) for x, y in loader])
    all_diffs = np.ones(n_examples, dtype=np.float32) * np.nan
    i_example = 0
    for x, y in loader:
        out = model_dist.model(x.cuda())[0]
        out_perturbation = th.rand_like(out)
        out_perturbation = 0.01 * out_perturbation / th.norm(out_perturbation,
                                                             p=2, dim=1,
                                                             keepdim=True)
        out_perturbed = out + out_perturbation
        inverted_perturbed = model_dist.model.invert(out_perturbed)[0]
        in_diffs = th.norm((x.cuda() - inverted_perturbed).view(x.shape[0], -1),
                           dim=1, p=2)
        assert not np.any(np.isnan(var_to_np(in_diffs)))
        all_diffs[i_example:i_example + len(x)] = var_to_np(in_diffs)
        i_example += len(x)
    assert i_example == len(all_diffs)
    assert not np.any(np.isnan(all_diffs)), "nan diff exists"

    return all_diffs


def get_out_diffs_per_set(loader, model_dist):
    n_examples = sum([len(x) for x, y in loader])
    all_diffs = np.ones(n_examples, dtype=np.float32) * np.nan
    i_example = 0
    for x, y in loader:
        x = x.cuda()
        in_perturbation = th.rand_like(x)
        in_perturbation = 0.01 * in_perturbation / th.norm(
            in_perturbation.view(x.shape[0],-1),
            p=2, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        in_perturbed = x + in_perturbation
        out = model_dist.model(x)[0]
        out_perturbed = model_dist.model(in_perturbed)[0]
        out_diffs = th.norm((out - out_perturbed).view(x.shape[0], -1),
                           dim=1, p=2)
        assert not np.any(np.isnan(var_to_np(out_diffs)))
        all_diffs[i_example:i_example + len(x)] = var_to_np(out_diffs)
        i_example += len(x)
    assert i_example == len(all_diffs)
    assert not np.any(np.isnan(all_diffs)), "nan diff exists"
    return all_diffs


def compute_auc_for_scores(scores_a, scores_b):
    auc = roc_auc_score(
        np.concatenate((np.zeros_like(scores_a),
                       np.ones_like(scores_b)),
                      axis=0),
        np.concatenate((scores_a,
                       scores_b,),
                      axis=0))
    return auc

def compute_aupr_for_scores(scores_a, scores_b):
    auc = average_precision_score(
        np.concatenate((np.zeros_like(scores_a),
                       np.ones_like(scores_b)),
                      axis=0),
        np.concatenate((scores_a,
                       scores_b,),
                      axis=0))
    return auc


def collect_for_dataloader(dataloader, step_fn, show_tqdm=False):
    with th.no_grad():
        all_outs = []
        if show_tqdm:
            dataloader = tqdm(dataloader)
        for batch in dataloader:
            outs = step_fn(*batch)
            all_outs.append(outs)
    return all_outs


def collect_for_loaders(name_to_loader,step_fn, show_tqdm=False):
    results = {}
    for name in name_to_loader:#
        result = collect_for_dataloader(name_to_loader[name], step_fn,
                                        show_tqdm=show_tqdm)
        try:
            result = np.concatenate(result)
        except:
            pass
        results[name] = result
    return results


def collect_log_dets(loader, model, node_names, use_y=False):
    results_model = IntermediateResultsNode(get_nodes_by_names(model, *node_names))
    def get_log_dets(x,y):
        # return in examples x modules logic
        if use_y:
            this_y = y
        else:
            this_y = None
        return np.array([var_to_np(logdet) for logdet in results_model(
            x,fixed=dict(y=this_y))[1]]).T
    log_dets = collect_for_dataloader(loader, get_log_dets, show_tqdm=True)
    log_dets_per_node = np.concatenate(log_dets, axis=0).T#np.array(log_dets).reshape(-1, log_dets[0].shape[-1]).T
    # now modules x examples
    name_to_log_det = dict(zip(node_names, log_dets_per_node))
    return name_to_log_det

def collect_log_dets_for_loaders(loaders, model, node_names, use_y=False):
    loaders_to_log_dets = dict([(name, collect_log_dets(loader, model, node_names,
                                                        use_y=use_y))
                                for name, loader in loaders.items()])
    return loaders_to_log_dets


def compute_bpds(dataloader, model, use_y, n_batches=None, show_tqdm=True):

    bpds = []
    if show_tqdm:
        dataloader = tqdm(dataloader)
    for i_batch, (x, y) in enumerate(dataloader):
        if not use_y:
            y = None
        fixed = dict(y=y)
        with th.no_grad():
            n_dims = np.prod(x.shape[1:])
            nll = -(model(x, fixed=fixed)[1] - np.log(256) * n_dims)
            bpd = nll / (n_dims * np.log(2))
            bpds.append(bpd)
        if n_batches is not None and i_batch >= (n_batches - 1):
            break

    return th.cat(bpds).cpu()


def identity(x):
    return x


def collect_outputs(loader, model, process_fn=identity, use_y=False):
    all_outputs = []
    with th.no_grad():
        for batch in loader:
            x = batch[0]
            if use_y:
                y = batch[1]
            else:
                y = None
            outputs = model(x, fixed=dict(y=y))
            outputs = process_fn(outputs)
            all_outputs.append(outputs)
    return all_outputs
