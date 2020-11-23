from invglow import folder_locations
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import os
from collections import OrderedDict



data_file = open(os.path.join(folder_locations.tiny_data, 'tiny_images.bin'), "rb")

def load_image(idx):
    data_file.seek(idx * 3072)
    data = data_file.read(3072)
    return np.frombuffer(data, dtype='uint8').reshape(32, 32, 3, order="F")

cifar10_idxs = [int(l) - 1 for l in open(os.path.join(folder_locations.tiny_data, '80mn_cifar_idxs.txt'), 'r').readlines()]
cifar100_idxs = [int(l) - 1 for l in open(os.path.join(folder_locations.tiny_data, 'cifar_indexes'), 'r').readlines()
                if int(l) != 0]



n_total_images = 79302017
metadata = loadmat(os.path.join(folder_locations.tiny_data, 'tiny_index.mat'))
words = [a[0]  for a in metadata['word'][0]]
num_imgs =  metadata['num_imgs'][0]
offset = metadata['offset'][0]
w_to_n = OrderedDict(zip(words, num_imgs))
w_to_o = OrderedDict(zip(words, offset))
number_indices = list(range(w_to_o['number'], w_to_o['number'] + w_to_n['offset']))

batch_starts = np.arange(0, n_total_images, 64)[:-1] # last one is an incomplete batch, drop it

rng = np.random.RandomState(20191203)
i_rand_starts = rng.choice(batch_starts, len(batch_starts), replace=False)

this_i_rand_starts = i_rand_starts[:24415]



folder = os.path.join(folder_locations.tiny_data, 'chunks/')
for i_start in tqdm(this_i_rand_starts):
    arrs = [load_image(i) for i in range(i_start, i_start+64)]
    np.save(os.path.join(folder, f"{i_start:08d}_{i_start+64:08d}.all.npy"), np.stack(arrs))

# Recheck

rng = np.random.RandomState(20191203)
i_rand_starts = rng.choice(batch_starts, len(batch_starts), replace=False)

this_i_rand_starts = i_rand_starts[:24415]

for i_start in tqdm(this_i_rand_starts):
    filename = os.path.join(folder, f"{i_start:08d}_{i_start+64:08d}.all.npy")
    assert os.path.exists(filename)


# no-cifar set
rng = np.random.RandomState(20191203)
i_rand_starts = rng.choice(batch_starts, len(batch_starts), replace=False)

all_cifar_idxs = np.sort(cifar10_idxs + cifar100_idxs)
set_cifar = set(all_cifar_idxs)
i_selected_rand_starts = []
for i_start in tqdm(i_rand_starts):
    if not any([i in set_cifar for i in range(i_start, i_start+ 64)]):
        i_selected_rand_starts.append(i_start)
this_i_rand_starts = i_selected_rand_starts[:24415]
folder = os.path.join(folder_locations.tiny_data, 'chunks/')
for i_start in tqdm(this_i_rand_starts):
    arrs = [load_image(i) for i in range(i_start, i_start+64)]
    np.save(os.path.join(folder, f"{i_start:08d}_{i_start+64:08d}.exclude_cifar.npy"), np.stack(arrs))
