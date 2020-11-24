# hierarchical_anomaly_detection

Pytorch implementation of the NeurIPS 2020 paper [Understanding anomaly detection with deep invertible networks through hierarchies of distributions and features](https://proceedings.neurips.cc/paper/2020/hash/f106b7f99d2cb30c3db1c3cc0fde9ccb-Abstract.html). The code allows the users to reproduce and extend the results reported in the study. Please cite the above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication. It will neither be maintained nor monitored in any way.

## Requirements

This is a Python3 codebase.

You will need some libraries:
- Pytorch
- ignite
- tensorboardX
- numpy
- scipy
- scikit-learn
- torchvision
- tqdm
- opencv (for Serra replication)

You also need to add this folder to your python path

## Data

You first have to set folder locations in invglow/folder_locations.py and download 80 Million Tiny Images, LSUN etc.

Also copy the supplied files 80mn_cifar_idxs.txt and cifar_indexes file to your tiny images folder (only needed if you want to test excluding cifar from tiny which we did not do in the main manuscript)

Then you need to run python invglow/create_tiny.py to create the tiny dataset

## Structure

invglow folder contains code for invertible network experiments.

main.py shows some examples of how code should be used to obtain results in manuscript

main.py shows one example of how code is run, you first will need to create a tiny model, and then use saved model this folder to further finetune on other datasets, similar to the invertible network main.py logic

## Pretrained models

We provide some pretrained models at:

https://osf.io/ces72/?view_only=cc58b057ac084d25862b2f5f7fc056df

## License

hierarchical_anomaly_detection is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.

For a list of other open source components included in hierarchical_anomaly_detection, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).




