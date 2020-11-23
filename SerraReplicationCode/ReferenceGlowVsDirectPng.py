#!/usr/bin/env python
# coding: utf-8

# # Installation

# You need:
# 
# * PyTorch
# * Torchvision
# * Scikit-Learn
# * OpenCV
# * Clone https://github.com/y0ast/Glow-PyTorch/tree/181daaffcd0f3561f08c32d5b3846874bcc0481a

# ## Define your folders

# In[1]:


import os

# Set your folder where you cloned the glow-repo linked above
glow_code_folder = os.path.join(os.environ['HOME'], 'code/glow-do-deep/glow_do_deep/')

# Set your folder where you downloaded pretrained glow model from 
# https://github.com/y0ast/Glow-PyTorch
# http://www.cs.ox.ac.uk/people/joost.vanamersfoort/glow.zip
output_folder = os.path.join(os.environ['HOME'], 'code/glow-do-deep/glow/')


# Set here path to your PyTorch-CIFAR10/SVHN datasets
cifar10_path =  os.path.join(os.environ['HOME'], 'data/pytorch-datasets/data/CIFAR10/')
svhn_path =  os.path.join(os.environ['HOME'], 'data/pytorch-datasets/data/SVHN/')


# ## Some imports

# In[2]:


import torch
from torchvision import datasets
import numpy as np
torch.backends.cudnn.benchmark = True

import json

# Load tqdm if available for progress bar
# otherwise just no progress bar
try:
    from tqdm.autonotebook import tqdm
except ModuleNotFoundError:
    def tqdm(x):
        return x


# ##  Load pretrained model

# In[3]:


os.chdir(glow_code_folder)
from model import Glow
model_name = 'glow_affine_coupling.pt'

with open(output_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)
    
print(hparams)
image_shape = (32,32,3)
num_classes = 10
model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model.load_state_dict(torch.load(output_folder + model_name))
model.set_actnorm_init()

device = torch.device("cuda")
model = model.to(device)

model = model.eval()


# ## Load datasets
# 
# Set `download=True` in case you don't have them yet.

# In[4]:



test_svhn = datasets.SVHN(
    svhn_path,
    split='test',
    download=False)


test_cifar10 = datasets.CIFAR10(
    cifar10_path,
    train=False,
    download=False)

pytorch_datasets = dict(test_cifar10=test_cifar10,
               test_svhn=test_svhn)

np_arrays = dict()
loaders = dict()
for name, dataset in pytorch_datasets.items():
    np_arr = np.stack([np.array(x) for x,y in dataset])
    np_arrays[name] = np_arr
    # Ensure we are working on exactly same data.
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(np_arr)),
            batch_size=512, drop_last=False)
    loaders[name] = loader

    


# ## Compute PNG BPDs

# In[5]:


import cv2
def create_png_bpds(np_im_arr,):
    all_bpds = []
    for i_file, a_x in enumerate(tqdm(np_im_arr)):
        # This code was written using an author reply to our mails
        # Use highest compression level (9)
        img_encoded = cv2.imencode('.png', a_x, [int(cv2.IMWRITE_PNG_COMPRESSION),9])[1]
        assert img_encoded.shape[1] == 1
        all_bpds.append((len(img_encoded) * 8)/np.prod(a_x.shape))
    return all_bpds


png_bpds = dict([
    (name, create_png_bpds(np_im_arr))
    for name, np_im_arr in np_arrays.items()])


# ## Compute BPDs of Glow Model

# In[6]:


def preprocess(x, ):
    # Preprocess from tensor with:
    # dim ordering: B,H,W,C
    # values: 0-255 
    # 
    # to tensor with
    # dim ordering: B,C,H,W
    # values: -0.5 to +0.5
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78
    x = x.permute(0,3,1,2)
    n_bits = 8
    n_bins = 2**n_bits
    x = x / n_bins - 0.5
    return x.cuda()


# In[7]:


def compute_glow_bpds(model, loader):
    all_bpds = []
    for x, in tqdm(loader):
        with torch.no_grad():
            preproced_x = preprocess(x)
            _, bpd, _ =  model(preproced_x)
        all_bpds.append(bpd.cpu().numpy())

    all_bpds = np.concatenate(all_bpds)
    return all_bpds


# In[8]:


glow_bpds = dict([
    (name, compute_glow_bpds(model, loader))
    for name, loader in loaders.items()])


# ## Results

# In[9]:


from sklearn.metrics import roc_auc_score
def compute_auc_for_s_scores(scores_ood, scores_itd):
    # Assumes scores a should be higher
    auc = roc_auc_score(
        np.concatenate((np.ones_like(scores_ood),
                       np.zeros_like(scores_itd)),
                      axis=0),
        np.concatenate((scores_ood,
                       scores_itd,),
                      axis=0))
    return auc


# We reach substantially different values for S-Score (78.4% vs 95.0%), see https://arxiv.org/pdf/1909.11480.pdf Supplementary D, p.14, Table 6.

# In[10]:


s_score_cifar10 = glow_bpds['test_cifar10'] - png_bpds['test_cifar10']
s_score_svhn = glow_bpds['test_svhn'] - png_bpds['test_svhn']

compute_auc_for_s_scores(s_score_svhn, s_score_cifar10)


# We reach similar numbers for PNG only (7.7% in paper vs 7.9% here)

# In[11]:


# We reach si
compute_auc_for_s_scores(png_bpds['test_svhn'], png_bpds['test_cifar10'])

