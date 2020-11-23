import logging

import torch as th

from invglow.exp import run_exp
from invglow.evaluate import evaluate_without_noise

log = logging.getLogger(__name__)

th.backends.cudnn.benchmark = True
default_args = dict(
    lr=5e-4,
    weight_decay=5e-5,
    first_n=None,
    exclude_cifar_from_tiny=False,
    output_dir='.',
    reinit=False,
    saved_optimizer_path=None,
    noise_factor=1/256.0,
    flow_coupling='affine',
    init_class_model = False,
    batch_size=64,
    augment=True,
    warmup_steps=None,
    n_epochs=250,
    np_th_seed=20200610
)

# to run properly, set debug to False
debug = True

dataset = 'tiny'
base_set_name = 'tiny'
tiny_grey = False
saved_base_model_path = None
saved_model_path = None
outlier_loss = None
outlier_weight = None
outlier_temperature = None
ood_set_name = None
add_full_label_loss = False
on_top_class_model_name = None
K = 32
local_patches = False
block_type = 'conv'
flow_permutation = 'invconv'
LU_decomposed=True


## Pretrain Tiny


trainer, model = run_exp(
    dataset=dataset,
    debug=debug,
    saved_base_model_path=saved_base_model_path,
    saved_model_path=saved_model_path,
    base_set_name=base_set_name,
    outlier_weight=outlier_weight,
    outlier_loss=outlier_loss,
    ood_set_name=ood_set_name,
    outlier_temperature=outlier_temperature,
    K=K,
    on_top_class_model_name=on_top_class_model_name,
    add_full_label_loss=add_full_label_loss,
    tiny_grey=tiny_grey,
    local_patches=local_patches,
    block_type=block_type,
    flow_permutation=flow_permutation,
    LU_decomposed=LU_decomposed,
    **default_args)


saved_base_model_path = './tiny_model.th'
th.save(model, saved_base_model_path)
del trainer, model

## Finetune

dataset = 'cifar10'
base_set_name = 'tiny'
tiny_grey = False
saved_base_model_path = './tiny_model.th'
saved_model_path = './tiny_model.th'
outlier_loss = None
outlier_temperature = None
outlier_weight = None
ood_set_name = 'svhn' # just for eval
add_full_label_loss = False
on_top_class_model_name = None
K = 32
local_patches = False
block_type = 'conv'
flow_permutation = 'invconv'
LU_decomposed = True

trainer, model = run_exp(
    dataset=dataset,
    debug=debug,
    saved_base_model_path=saved_base_model_path,
    saved_model_path=saved_model_path,
    base_set_name=base_set_name,
    outlier_weight=outlier_weight,
    outlier_loss=outlier_loss,
    ood_set_name=ood_set_name,
    outlier_temperature=outlier_temperature,
    K=K,
    on_top_class_model_name=on_top_class_model_name,
    add_full_label_loss=add_full_label_loss,
    tiny_grey=tiny_grey,
    local_patches=local_patches,
    block_type=block_type,
    flow_permutation=flow_permutation,
    LU_decomposed=LU_decomposed,
    **default_args)
# We create some helper function to show how the evaluation wihtout noise works,
# please look inside to see how it works,
# this function was not used directly for the manuscript
# but should yield same results, unless new bugs were introduced :)

base_model = th.load(saved_base_model_path)

evaluate_without_noise(model, base_model, on_top_class=False,
                      first_n=512, # set to none for proper eval
                       noise_factor=1/256.0,
                       in_dist_name='cifar10',
                       rgb_or_grey='rgb',
                       only_full_nll=False)
del trainer, model


## Finetune with outlier loss

dataset = 'cifar10'
base_set_name = 'tiny'
tiny_grey = False
saved_base_model_path = './tiny_model.th'
saved_model_path = './tiny_model.th'
outlier_loss = 'class'
outlier_weight = 6000
outlier_temperature = 1000
ood_set_name = 'svhn' # just for eval
add_full_label_loss = False
on_top_class_model_name = None
K = 32
local_patches = False
block_type = 'conv'
flow_permutation = 'invconv'
LU_decomposed = True

trainer, model = run_exp(
    dataset=dataset,
    debug=debug,
    saved_base_model_path=saved_base_model_path,
    saved_model_path=saved_model_path,
    base_set_name=base_set_name,
    outlier_weight=outlier_weight,
    outlier_loss=outlier_loss,
    ood_set_name=ood_set_name,
    outlier_temperature=outlier_temperature,
    K=K,
    on_top_class_model_name=on_top_class_model_name,
    add_full_label_loss=add_full_label_loss,
    tiny_grey=tiny_grey,
    local_patches=local_patches,
    block_type=block_type,
    flow_permutation=flow_permutation,
    LU_decomposed=LU_decomposed,
    **default_args)


## Finetune with outlier loss supervised

del trainer, model

dataset = 'cifar10'
base_set_name = 'tiny'
tiny_grey = False
saved_base_model_path = './tiny_model.th'
saved_model_path = './tiny_model.th'
outlier_loss = 'class'
outlier_weight = 6000
outlier_temperature = 1000
ood_set_name = 'svhn' # just for eval
add_full_label_loss = True
on_top_class_model_name = 'latent'
K = 32
local_patches = False
block_type = 'conv'
flow_permutation = 'invconv'
LU_decomposed = True

trainer, model = run_exp(
    dataset=dataset,
    debug=debug,
    saved_base_model_path=saved_base_model_path,
    saved_model_path=saved_model_path,
    base_set_name=base_set_name,
    outlier_weight=outlier_weight,
    outlier_loss=outlier_loss,
    ood_set_name=ood_set_name,
    outlier_temperature=outlier_temperature,
    K=K,
    on_top_class_model_name=on_top_class_model_name,
    add_full_label_loss=add_full_label_loss,
    tiny_grey=tiny_grey,
    local_patches=local_patches,
    block_type=block_type,
    flow_permutation=flow_permutation,
    LU_decomposed=LU_decomposed,
    **default_args)

## Local model

del trainer, model

dataset = 'cifar10'
base_set_name = None
tiny_grey = False
saved_base_model_path = None
saved_model_path = None
outlier_loss = None
outlier_weight = None
outlier_temperature = None
ood_set_name = None
add_full_label_loss = False
on_top_class_model_name = None
K = 32
local_patches = True
block_type = 'conv'
flow_permutation = 'invconv'
LU_decomposed = True

trainer, model = run_exp(
    dataset=dataset,
    debug=debug,
    saved_base_model_path=saved_base_model_path,
    saved_model_path=saved_model_path,
    base_set_name=base_set_name,
    outlier_weight=outlier_weight,
    outlier_loss=outlier_loss,
    ood_set_name=ood_set_name,
    outlier_temperature=outlier_temperature,
    K=K,
    on_top_class_model_name=on_top_class_model_name,
    add_full_label_loss=add_full_label_loss,
    tiny_grey=tiny_grey,
    local_patches=local_patches,
    block_type=block_type,
    flow_permutation=flow_permutation,
    LU_decomposed=LU_decomposed,
    **default_args)

## Fully Connected Model

del trainer, model

dataset = 'cifar10'
base_set_name = None
tiny_grey = False
saved_base_model_path = None
saved_model_path = None
outlier_loss = None
outlier_weight = None
outlier_temperature = None
ood_set_name = None
add_full_label_loss = False
on_top_class_model_name = None
K = 8
local_patches = False
block_type = 'dense'
flow_permutation = 'invconvfixed'
LU_decomposed = False

trainer, model = run_exp(
    dataset=dataset,
    debug=debug,
    saved_base_model_path=saved_base_model_path,
    saved_model_path=saved_model_path,
    base_set_name=base_set_name,
    outlier_weight=outlier_weight,
    outlier_loss=outlier_loss,
    ood_set_name=ood_set_name,
    outlier_temperature=outlier_temperature,
    K=K,
    on_top_class_model_name=on_top_class_model_name,
    add_full_label_loss=add_full_label_loss,
    tiny_grey=tiny_grey,
    local_patches=local_patches,
    block_type=block_type,
    flow_permutation=flow_permutation,
    LU_decomposed=LU_decomposed,
    **default_args)