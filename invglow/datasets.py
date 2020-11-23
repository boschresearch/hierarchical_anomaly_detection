import os
import os.path
from collections.abc import Iterable
from glob import glob
from pathlib import Path

import PIL
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets
from torchvision.datasets.lsun import LSUNClass
from torchvision.datasets.utils import verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset

from invglow import folder_locations
from invglow.util import np_to_var


class PreprocessedLoader(object):
    def __init__(self, dataloader, module, to_cuda):
        self.dataloader = dataloader
        self.module = module
        self.to_cuda = to_cuda

    def __iter__(self):
        for batch in self.dataloader:
            # First convert x, check if x is a tuple (e.g., multiple transforms
            # applied to same batch)
            x = batch[0]
            if hasattr(x, 'cuda'):
                x = x.cuda()
                x = self.module(x)
            else:
                preproced_xs = []
                for a_x in x:
                    a_x = a_x.cuda()
                    a_x = self.module(a_x)
                    preproced_xs.append(a_x)
                x = tuple(preproced_xs)
            remaining_batch = batch[1:]
            if self.to_cuda:
                remaining_batch = tuple([a.cuda() for a in remaining_batch])
            yield (x,) + remaining_batch

    def __len__(self):
        return len(self.dataloader)



def preprocess(x, n_bits=8):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2**n_bits
    if n_bits < 8:
      x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x


def postprocess(x, n_bits=8):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2**n_bits
    return torch.clamp(x, 0, 255).byte()


def get_one_hot_encode(num_classes):
    def target_to_one_to_hot(target):
        return F.one_hot(torch.tensor(target), num_classes)
    return target_to_one_to_hot


def get_CIFAR10(augment, dataroot, download, first_n):
    image_shape = (32, 32, 3)
    num_classes = 10

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])
    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])

    train_transform = transforms.Compose(transformations)

    one_hot_encode = get_one_hot_encode(num_classes)
    path = Path(dataroot) / 'data' / 'CIFAR10'
    train_dataset = datasets.CIFAR10(path, train=True,
                                     transform=train_transform,
                                     target_transform=one_hot_encode,
                                     download=download)
    if first_n is not None:
        train_dataset.data = train_dataset.data[:first_n]
        train_dataset.targets = train_dataset.targets[:first_n]
    test_dataset = datasets.CIFAR10(path, train=False,
                                    transform=test_transform,
                                    target_transform=one_hot_encode,
                                    download=download)
    if first_n is not None:
        test_dataset.data = test_dataset.data[:first_n]
        test_dataset.targets = test_dataset.targets[:first_n]

    return image_shape, num_classes, train_dataset, test_dataset


def get_CIFAR100(augment, dataroot, download, first_n):
    image_shape = (32, 32, 3)
    num_classes = 100

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])

    train_transform = transforms.Compose(transformations)

    one_hot_encode = get_one_hot_encode(num_classes)

    #path = Path(dataroot) / 'data' / 'CIFAR100'
    path = dataroot # hack
    train_dataset = datasets.CIFAR100(path, train=True,
                                     transform=train_transform,
                                     target_transform=one_hot_encode,
                                     download=download)
    if first_n is not None:
        train_dataset.data = train_dataset.data[:first_n]
        train_dataset.targets = train_dataset.targets[:first_n]
    test_dataset = datasets.CIFAR100(path, train=False,
                                    transform=test_transform,
                                    target_transform=one_hot_encode,
                                    download=download)
    if first_n is not None:
        test_dataset.data = test_dataset.data[:first_n]
        test_dataset.targets = test_dataset.targets[:first_n]

    return image_shape, num_classes, train_dataset, test_dataset


def get_SVHN(augment, dataroot, download, first_n):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])
    one_hot_encode = get_one_hot_encode(num_classes)

    path = Path(dataroot) / 'data' / 'SVHN'
    train_dataset = datasets.SVHN(path, split='train',
                                  transform=train_transform,
                                  target_transform=one_hot_encode,
                                  download=download)
    if first_n is not None:
        train_dataset.data = train_dataset.data[:first_n]
        train_dataset.labels = train_dataset.labels[:first_n]

    test_dataset = datasets.SVHN(path, split='test',
                                 transform=test_transform,
                                 target_transform=one_hot_encode,
                                 download=download)
    if first_n is not None:
        test_dataset.data = test_dataset.data[:first_n]
        test_dataset.labels = test_dataset.labels[:first_n]

    return image_shape, num_classes, train_dataset, test_dataset


class TensorDatasetWithTransforms(th.utils.data.TensorDataset):
    # adapted from https://stackoverflow.com/a/55593757/1469195
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        result = [x,y]
        for other_tensors in self.tensors[2:]:
            result.append(other_tensors[index])

        return tuple(result)

    def __len__(self):
        return self.tensors[0].size(0)


def load_train_test_with_defaults(
        dataset,
        first_n,
        shuffle_train=True,
        drop_last_train=True,
        batch_size=64,
        eval_batch_size=512,
        n_workers=2,
        augment=True,
        exclude_cifar_from_tiny=False,
        shuffle_tiny_chunks=True,
        tiny_grey=False,
        resize_tiny=None):
    return load_train_test(
        dataset=dataset,
        shuffle_train=shuffle_train,
        drop_last_train=drop_last_train,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        n_workers=n_workers,
        first_n=first_n,
        augment=augment,
        exclude_cifar_from_tiny=exclude_cifar_from_tiny,
        shuffle_tiny_chunks=shuffle_tiny_chunks,
        tiny_grey=tiny_grey,
        resize_tiny=resize_tiny,
        )


def load_train_test(dataset, shuffle_train, drop_last_train,
                    batch_size, eval_batch_size,
                    n_workers,
                    first_n, augment, exclude_cifar_from_tiny,
                    dataroot=folder_locations.pytorch_data,
                    shuffle_tiny_chunks=True,
                    tiny_grey=False,
                    resize_tiny=None):
    assert dataset in ['tiny', 'svhn', 'cifar10', 'cifar100',
                       'tinyimagenet', 'random_tiny', 'mnist',
                       'fashion-mnist',
                       ] + ['brats_' + s for s in ['all', 't1', 't1ce', 't2', 'flair']], (
        f"Unknown dataset {dataset}"
    )
    if dataset == 'tiny':
        assert exclude_cifar_from_tiny is not None
        return load_train_test_tiny(
            n_images=first_n, shuffle_train=shuffle_train,
            drop_last_train=drop_last_train,
            batch_size=batch_size, eval_batch_size=eval_batch_size,
            augment=augment, exclude_cifar=exclude_cifar_from_tiny,
            shuffle_chunks=shuffle_tiny_chunks,
            tiny_grey=tiny_grey,
            resize_tiny=resize_tiny,)
    elif 'brats' in dataset:
        pattern = dataset.split('_')[1]
        return load_brats(
            first_n=first_n, batch_size=batch_size,
            n_workers=n_workers, eval_batch_size=eval_batch_size,
            pattern=pattern,
            shuffle_train=shuffle_train, drop_last_train=drop_last_train)
    else:
        assert dataset in ['svhn', 'cifar10', 'cifar100', 'mnist',
                           'fashion-mnist']
        return load_train_test_as_glow(
            dataset, dataroot=dataroot,
            first_n=first_n, shuffle_train=shuffle_train,
            drop_last_train=drop_last_train,
            batch_size=batch_size, eval_batch_size=eval_batch_size,
            augment=augment, download=False, n_workers=n_workers)


def load_train_test_tiny(
        n_images, batch_size, eval_batch_size,
        exclude_cifar,
        shuffle_chunks=True,
        shuffle_train=True, drop_last_train=True,
        augment=False, tiny_grey=False, resize_tiny=None):
    tiny_folder = folder_locations.tiny_data
    if exclude_cifar:
        chunk_files = sorted(glob(
            os.path.join(tiny_folder, 'chunks/*.exclude_cifar.npy')))
    else:
        assert not exclude_cifar
        chunk_files = sorted(glob(os.path.join(
            tiny_folder,'chunks/*.all.npy')))

    chunk_files = np.array(chunk_files)
    if n_images is None:
        n_chunks = len(chunk_files)
    else:
        n_chunks = n_images // 64
    if shuffle_chunks:
        chosen_chunks = np.random.choice(len(chunk_files), n_chunks, replace=False)
    else:
        chosen_chunks = np.arange(n_chunks)
    np_arr = np.concatenate([np.load(f) for f in chunk_files[chosen_chunks]])
    if tiny_grey:
        # https://stackoverflow.com/q/687261/1469195
        w = np.array([0.2989, 0.5870, 0.1140])
        np_arr = np.sum(np_arr * w[None, None, None], axis=-1, keepdims=True)

    tiny_var = np_to_var(np_arr.astype(np.float32).transpose(0,3,1,2) / 255 - 0.5)
    del np_arr
    n_examples = len(tiny_var)
    # to make preprocessing identical to CIFAR/SVHN
    tiny_var = (tiny_var+0.5) * (255/256) - 0.5

    n_batches = n_examples / batch_size
    n_train = int(np.round(n_batches * 0.8) * batch_size)
    train = tiny_var[:n_train]
    test = tiny_var[n_train:]
    def to_image(x):
        # undo rescaling, and make into PIL image
        x_arr = ((x.cpu().numpy().transpose(1,2,0) + 0.5) * 256).astype(np.uint8)
        if tiny_grey:
            x_arr = x_arr.squeeze(-1)
        return Image.fromarray(x_arr)

    train_image_transforms = []
    common_image_transforms = []
    if augment:
        train_image_transforms.extend([
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),])
    if resize_tiny is not None:
        common_image_transforms.append(
            transforms.Resize((resize_tiny, resize_tiny), interpolation=Image.BILINEAR),)

    if (len(train_image_transforms) + len(common_image_transforms)) > 0:
        train_transforms = [to_image] + train_image_transforms +  (
                common_image_transforms + [transforms.ToTensor(), preprocess])
        train_transformations = transforms.Compose(train_transforms)
    else:
        train_transformations = None
    if len(common_image_transforms) > 0:
        test_transforms = [to_image] + (
                common_image_transforms + [transforms.ToTensor(), preprocess])
        test_transformations = transforms.Compose(test_transforms)
    else:
        test_transformations = None

    # random ys
    train_set = TensorDatasetWithTransforms(
        train, F.one_hot(th.ones(len(train), dtype=th.int64), num_classes=10),
        transform=train_transformations)
    train_loader = th.utils.data.DataLoader(train_set,
                             batch_size=batch_size,
                             shuffle=shuffle_train, num_workers=0,
                             drop_last=drop_last_train)
    test_set = TensorDatasetWithTransforms(
        test,F.one_hot(th.ones(len(test), dtype=th.int64), num_classes=10),
        transform=test_transformations)
    test_loader = th.utils.data.DataLoader(test_set,
                             batch_size=eval_batch_size,
                             shuffle=False, num_workers=0,
                             drop_last=False)
    return train_loader, test_loader


def check_dataset(dataset, dataroot, augment, download, first_n):
    if dataset == 'cifar10':
        cifar10 = get_CIFAR10(augment, dataroot, download, first_n=first_n)
        input_size, num_classes, train_dataset, test_dataset = cifar10
    if dataset == 'cifar100':
        cifar100 = get_CIFAR100(augment, dataroot, download, first_n=first_n)
        input_size, num_classes, train_dataset, test_dataset = cifar100
    if dataset == 'svhn':
        svhn = get_SVHN(augment, dataroot, download, first_n=first_n)
        input_size, num_classes, train_dataset, test_dataset = svhn
    if dataset == 'mnist':
        mnist = get_MNIST(augment, dataroot, download, first_n=first_n,
                         rescale_to_32=True)
        input_size, num_classes, train_dataset, test_dataset = mnist
    if dataset == 'fashion-mnist':
        mnist = get_FASHION_MNIST(augment, dataroot, download, first_n=first_n,
                         rescale_to_32=True)
        input_size, num_classes, train_dataset, test_dataset = mnist
    return input_size, num_classes, train_dataset, test_dataset


def get_MNIST(augment, dataroot, download, first_n, rescale_to_32):
    image_shape = (28, 28, 1)

    if rescale_to_32:
        image_shape = (32, 32, 1)

    num_classes = 10
    common_transformations = []
    if rescale_to_32:
        common_transformations.append(
            transforms.Resize((32, 32), interpolation=PIL.Image.BILINEAR))

    # do like that to get list copy
    train_transformations = [] + common_transformations
    if augment:
        train_transformations.append(
            transforms.RandomAffine(0, translate=(0.1, 0.1)))

    train_transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(train_transformations)

    test_transform = transforms.Compose(
        common_transformations + [transforms.ToTensor(), preprocess])
    one_hot_encode = get_one_hot_encode(num_classes)

    path = Path(dataroot)
    train_dataset = datasets.MNIST(
        path, train=True, transform=train_transform,
        target_transform=one_hot_encode,download = download)
    if first_n is not None:
        train_dataset.data = train_dataset.data[:first_n]
        train_dataset.targets = train_dataset.targets[:first_n]

    test_dataset = datasets.MNIST(path, train=False,
                                  transform=test_transform,
                                  target_transform=one_hot_encode,
                                  download=download)
    if first_n is not None:
        test_dataset.data = test_dataset.data[:first_n]
        test_dataset.targets = test_dataset.targets[:first_n]

    return image_shape, num_classes, train_dataset, test_dataset


def get_FASHION_MNIST(augment, dataroot, download, first_n, rescale_to_32):
    image_shape = (28, 28, 1)

    if rescale_to_32:
        image_shape = (32, 32, 1)

    num_classes = 10
    common_transformations = []
    if rescale_to_32:
        common_transformations.append(
            transforms.Resize((32, 32), interpolation=PIL.Image.BILINEAR))

    # do like that to get list copy
    train_transformations = [] + common_transformations
    if augment:
        train_transformations.append(
            transforms.RandomAffine(0, translate=(0.1, 0.1)))

    train_transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(train_transformations)

    test_transform = transforms.Compose(
        common_transformations + [transforms.ToTensor(), preprocess])
    one_hot_encode = get_one_hot_encode(num_classes)

    path = Path(dataroot)
    train_dataset = datasets.FashionMNIST(
        path, train=True, transform=train_transform,
        target_transform=one_hot_encode,download = download)
    if first_n is not None:
        train_dataset.data = train_dataset.data[:first_n]
        train_dataset.targets = train_dataset.targets[:first_n]

    test_dataset = datasets.FashionMNIST(path, train=False,
                                  transform=test_transform,
                                  target_transform=one_hot_encode,
                                  download=download)
    if first_n is not None:
        test_dataset.data = test_dataset.data[:first_n]
        test_dataset.targets = test_dataset.targets[:first_n]

    return image_shape, num_classes, train_dataset, test_dataset


def load_train_test_as_glow(dataset, dataroot, augment, download, first_n,
                            batch_size, n_workers, eval_batch_size,
                            shuffle_train=True, drop_last_train=True):
    ds = check_dataset(dataset, dataroot, augment, download, first_n=first_n)
    image_shape, num_classes, train_dataset, test_dataset = ds
    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=shuffle_train, num_workers=n_workers,
                                   drop_last=drop_last_train)
    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size,
                                  shuffle=False, num_workers=n_workers,
                                  drop_last=False)
    return train_loader, test_loader


class BRATS2018(th.utils.data.Dataset):
    def __init__(self, root_dir, n_images, train, pattern, transform=None, ):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.n_images = n_images
        self.train = train
        if train:
            folders = [os.path.join(self.root_dir, 'train', s) for s in
                       ['lgg', 'hgg']]
        else:
            folders = [os.path.join(self.root_dir, 'valid')]
        self.filenames = []
        for folder in folders:
            if pattern == 'all':
                this_filenames = glob(os.path.join(folder, '*.jpg'))
            else:
                assert pattern in ['t1', 't2', 't1ce', 'flair']
                this_filenames = glob(
                    os.path.join(folder, f'*_{pattern}.*.jpg'))
            self.filenames.extend(this_filenames)
        if n_images is not None:
            self.filenames = self.filenames[:n_images]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        image = Image.open(img_name)
        if self.transform is not None:
            image = self.transform(image)

        return image, th.ones(1)


def load_brats(first_n, batch_size, n_workers, eval_batch_size,
               pattern,
               shuffle_train=True, drop_last_train=True):
    brats_folder = folder_locations.brats_data
    image_folder = os.path.join(brats_folder, 'jpgs/')

    def to_rgb(im):
        return im.repeat(3, 1, 1)

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.ToTensor(), preprocess, to_rgb])
    test_transforms = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(), preprocess, to_rgb])
    train_dataset = BRATS2018(image_folder, n_images=first_n, train=True,
                              pattern=pattern, transform=train_transforms)
    test_dataset = BRATS2018(image_folder, n_images=first_n, train=False,
                             pattern=pattern, transform=test_transforms)

    train_loader = th.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle_train,
                                            num_workers=n_workers,
                                            drop_last=drop_last_train)
    test_loader = th.utils.data.DataLoader(test_dataset,
                                           batch_size=eval_batch_size,
                                           shuffle=False, num_workers=n_workers,
                                           drop_last=False)
    return train_loader, test_loader


class CelebA(th.utils.data.Dataset):
    def __init__(self,root_dir, n_images, transform=None, ):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.n_images = n_images

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'celeb_{idx:d}.png')
        image = Image.open(img_name)
        if self.transform is not None:
            image = self.transform(image)

        return image, th.ones(1)



def load_celeb_a(first_n, batch_size, n_workers, eval_batch_size,
                shuffle_train=True, drop_last_train=True):
    image_folder = folder_locations.celeba_data
    train_dataset, test_dataset = get_celeb_a(image_folder, first_n)
    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=shuffle_train, num_workers=n_workers,
                                   drop_last=drop_last_train)
    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size,
                                  shuffle=False, num_workers=n_workers,
                                  drop_last=False)
    return train_loader, test_loader

def get_celeb_a(image_folder, first_n):
    if first_n is None:
        first_n = 120000
    transformations = transforms.Compose([transforms.ToTensor(), preprocess])
    dataset = CelebA(image_folder, first_n, transform=transformations,)
    train_set = th.utils.data.Subset(dataset, range(0, int(0.8*first_n)))
    test_set = th.utils.data.Subset(dataset, range(int(0.8*first_n), first_n))

    return train_set, test_set


class TinyImageNet(th.utils.data.Dataset):
    def __init__(self, root_dir, n_images, train_or_test, transform=None, ):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.n_images = n_images
        self.train_or_test = train_or_test
        if self.train_or_test == 'train':
            folds = ('train', 'val',)
        else:
            assert self.train_or_test == 'test'
            folds = ('test',)

        all_image_files = []
        for fold in folds:
            if fold == 'train':
                subfolders = glob(os.path.join(self.root_dir, fold, '*'))
                for f in subfolders:
                    image_files = glob(os.path.join(f, 'images/', '*.JPEG'))
                    all_image_files.extend(image_files)
            else:
                image_files = glob(os.path.join(self.root_dir, fold, 'images', '*.JPEG'))
                all_image_files.extend(image_files)
        self.all_image_files = all_image_files
        if self.n_images is not None:
            self.all_image_files = self.all_image_files[:self.n_images]


    def __len__(self):
        return len(self.all_image_files)

    def __getitem__(self, idx):
        img_name = self.all_image_files[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, th.ones(1)


def load_tiny_imagenet(first_n, batch_size, n_workers, eval_batch_size,
                shuffle_train=True, drop_last_train=True):
    image_folder = folder_locations.tiny_imagenet_data

    train_dataset, test_dataset = get_tiny_imagenet(image_folder, first_n)
    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=shuffle_train, num_workers=n_workers,
                                   drop_last=drop_last_train)
    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size,
                                  shuffle=False, num_workers=n_workers,
                                  drop_last=False)
    return train_loader, test_loader


def get_tiny_imagenet(image_folder, first_n):
    transformations = transforms.Compose(
        [transforms.Resize((32, 32), interpolation=Image.BILINEAR),
         transforms.ToTensor(), preprocess])
    train_set = TinyImageNet(
        image_folder, n_images=first_n,
        train_or_test='train', transform=transformations)
    test_set = TinyImageNet(
        image_folder, n_images=first_n,
        train_or_test='test', transform=transformations)
    return train_set, test_set



class LSUN(VisionDataset):
    """
    `LSUN <https://www.yf.io/p/lsun>`_ dataset.
    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, classes='train', transform=None, target_transform=None):
        super(LSUN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.classes = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(
                root=root + '/' + c + '_lmdb',
                transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes):
        categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                      'conference_room', 'dining_room', 'kitchen',
                      'living_room', 'restaurant', 'tower']
        dset_opts = ['train', 'val', 'test']

        try:
            verify_str_arg(classes, "classes", dset_opts)
            if classes == 'test':
                classes = [classes]
            else:
                classes = [c + '_' + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = ("Expected type str or Iterable for argument classes, "
                       "but got type {}.")
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            #msg_fmtstr = ("Expected type str for elements in argument classes, "
            #              "but got type {}.")
            for c in classes:
                #print("the c !!", c)
                msg_fmtstr = ("Expected type str for elements in argument classes, "
                              "but got type {}.")
                verify_str_arg(c, custom_msg=msg_fmtstr.format(type(c)))
                c_short = c.split('_')
                category, dset_opt = '_'.join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class",
                                        iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self):
        return self.length

    def extra_repr(self):
        return "Classes: {classes}".format(**self.__dict__)


