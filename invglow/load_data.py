from invglow.datasets import load_train_test, PreprocessedLoader
import torch as th

from invglow.invertible.noise import UniNoise
from invglow.invertible.pure_model import NoLogDet
from invglow.datasets import LSUN
from torchvision import transforms
from invglow import folder_locations

def load_data(
        dataset,
        first_n,
        exclude_cifar_from_tiny,
        base_set_name,
        ood_set_name,
        noise_factor,
        augment,
        batch_size=64,
        eval_batch_size=512,
        shuffle_train=True,
        drop_last_train=True,
        tiny_grey=False,
        ):
    n_workers = 6

    base_first_n = first_n
    train_loader, test_loader = load_train_test(
        dataset,
        shuffle_train=shuffle_train,
        drop_last_train=drop_last_train,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        n_workers=n_workers,
        first_n=first_n,
        augment=augment,
        exclude_cifar_from_tiny=exclude_cifar_from_tiny,
        tiny_grey=tiny_grey)

    if base_set_name is not None:
        base_train_loader, base_test_loader = load_train_test(
            base_set_name,
            shuffle_train=shuffle_train,
            drop_last_train=drop_last_train,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            n_workers=n_workers,
            first_n=base_first_n,
            augment=augment,
            exclude_cifar_from_tiny=exclude_cifar_from_tiny,
            tiny_grey=tiny_grey)
    else:
        base_train_loader = None
    if ood_set_name is not None:
        ood_train_loader, ood_test_loader = load_train_test(
            ood_set_name,
            shuffle_train=shuffle_train,
            drop_last_train=drop_last_train,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            n_workers=n_workers,
            first_n=first_n,
            augment=augment,
            exclude_cifar_from_tiny=exclude_cifar_from_tiny,
            tiny_grey=tiny_grey)

        if tiny_grey == False:
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
    def preproced(loader):
        return PreprocessedLoader(loader, NoLogDet(UniNoise(noise_factor)),
                                  to_cuda=True)
    loaders = dict(
        train=preproced(train_loader),
        test=preproced(test_loader),
    )
    if ood_set_name is not None:
        loaders['ood_test'] = preproced(ood_test_loader)
        if tiny_grey == False:
            loaders['lsun'] = preproced(test_lsun)
            if dataset == 'cifar10':
                other_cifar_name = 'cifar100'
            else:
                other_cifar_name = 'cifar10'
            _, other_cifar_test_loader = load_train_test(
                other_cifar_name,
                shuffle_train=shuffle_train,
                drop_last_train=drop_last_train,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                n_workers=n_workers,
                first_n=first_n,
                augment=augment,
                exclude_cifar_from_tiny=exclude_cifar_from_tiny,
                tiny_grey=tiny_grey)
            loaders['ood_cifar'] = preproced(other_cifar_test_loader)


    if base_set_name is not None:
        base_train_loader = preproced(base_train_loader)

    return loaders, base_train_loader
