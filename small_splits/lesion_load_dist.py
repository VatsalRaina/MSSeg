"""
Plot performance of each model against lesion load (percentage lesion voxels) of patients.
"""

import argparse
import os
import torch
import sys
import re
from glob import glob
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss
from monai.metrics import compute_meandice, DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,ToTensord,ConcatItemsd,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, AsDiscrete, Activations)
from monai.utils import first, set_determinism
from monai.data import write_nifti, create_file_basename, NiftiDataset
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import pandas as pd

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--path_gts_train', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_gts_dev', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_gts_test', type=str, default='', help='Specify the path to the test gts directory')

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    segs_train = sorted(glob(os.path.join(args.path_gts_train, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))  
    N = (len(segs_train)) 
    indices = np.arange(N)
    v=indices[:]
    files_train=[]
    for j in v:
        files_train = files_train + [{"label": seg} for seg in segs_train[j:j+1]]
   
    segs_dev = sorted(glob(os.path.join(args.path_gts_dev, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))  
    N = (len(segs_dev)) 
    indices = np.arange(N)
    v=indices[:]
    files_dev=[]
    for j in v:
        files_dev = files_dev + [{"label": seg} for seg in segs_dev[j:j+1]]

    segs_test = sorted(glob(os.path.join(args.path_gts_test, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))  
    N = (len(segs_test)) 
    indices = np.arange(N)
    v=indices[:]
    files_test=[]
    for j in v:
        files_test = files_test + [{"label": seg} for seg in segs_test[j:j+1]]

    transforms = Compose(
    [
        LoadNiftid(keys=["label"]),
        AddChanneld(keys=["label"]),
        Spacingd(keys=["label"], pixdim=(1.0, 1.0, 1.0), mode=("nearest")),
        ToTensord(keys=["label"]),
    ]
    )

    ds_train = CacheDataset(data=files_train, transform=transforms, cache_rate=0.5, num_workers=0)
    loader_train = DataLoader(ds_train, batch_size=1, num_workers=0)

    ds_dev = CacheDataset(data=files_dev, transform=transforms, cache_rate=0.5, num_workers=0)
    loader_dev = DataLoader(ds_dev, batch_size=1, num_workers=0)

    ds_test = CacheDataset(data=files_test, transform=transforms, cache_rate=0.5, num_workers=0)
    loader_test = DataLoader(ds_test, batch_size=1, num_workers=0)

    train_load = []
    dev_load = []
    test_load = []

    with torch.no_grad():
        for count, batch_data in enumerate(loader_train):
            print(count)
            gt  = (batch_data["label"].type(torch.LongTensor).to(device))
            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)
            load = np.sum(gt) / len(gt.flatten())
            train_load.append(load)
        for count, batch_data in enumerate(loader_dev):
            print(count)
            gt  = (batch_data["label"].type(torch.LongTensor).to(device))
            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)
            load = np.sum(gt) / len(gt.flatten())
            dev_load.append(load)
        for count, batch_data in enumerate(loader_test):
            print(count)
            gt  = (batch_data["label"].type(torch.LongTensor).to(device))
            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)
            load = np.sum(gt) / len(gt.flatten())
            test_load.append(load)


 
    sns.scatterplot(x=lesion_size_min, y=model1_dsc_norm, label="Min")
    sns.scatterplot(x=lesion_size_mean, y=model1_dsc_norm, label="Mean")
    sns.scatterplot(x=lesion_size_max, y=model1_dsc_norm, label="Max")
    plt.xscale('log')
    plt.xlabel(r"Lesion Size ($mm^3$)")
    plt.ylabel(r"$\overline{DSC}$")
    plt.ylim([-0.03, 1.0])
    plt.legend()
    plt.savefig('lesionSize.png', bbox_inches="tight")
    plt.clf()


#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)