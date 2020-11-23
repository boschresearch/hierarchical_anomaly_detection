import os

# Just set your paths here explicitly, no need
# to use system environment as we did, this just made it easier for us

pytorch_data = os.environ['pytorch_data']
# Also copy the 80mn_cifar_idxs.txt and cifar_indexes file there
tiny_data = os.environ['tiny_data']
lsun_data = os.environ['lsun_data']
# only necessary for MRI experiment:
brats_data = os.environ['brats_data']
# only necessary for additional OOD dataset evaluation:
celeba_data = os.environ['celeba_data']
tiny_imagenet_data = os.environ['tiny_imagenet_data']