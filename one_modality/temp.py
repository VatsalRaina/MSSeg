"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from glob import glob
import torch
import re
from torch import nn
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss
from monai.metrics import compute_meandice
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,ToTensord,ConcatItemsd,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, RandScaleIntensityd, Activations,SqueezeDimd)
from monai.utils import first
import numpy as np
import random

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify the initial learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the training data files directory')
parser.add_argument('--path_save', type=str, default='', help='Specify the path to the trained model will be saved')

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

    device = get_default_device()

    root_dir = args.path_save  # Path where the trained model is located
    path_data = args.path_data
    flair = sorted(glob(os.path.join(path_data, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(path_data, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))                   # Collect all corresponding ground truths

    N = (len(flair)) # Number of subjects for training/validation, by default using all subjects in the folder
    
    #indices = np.random.permutation(N)
    indices = np.asarray([3, 7, 6, 2, 10, 4, 1, 13, 0, 14, 9, 8, 12, 11, 5])

    #indices = np.arange(N)
    # v=indices[:]
    v=indices[:3] 

    val_files=[]
    for j in v:
        val_files = val_files + [{"image": fl,"label": seg} for fl, seg in zip(flair[j:j+1], segs[j:j+1])]
    print("Validation cases:", len(val_files))

    val_transforms = Compose(
    [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image","label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "label"]),
    ]
    )

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=0).to(device)

    model.load_state_dict(torch.load(os.path.join(root_dir, "Best_model_finetuning.pth")))
    
    act = Activations(softmax=True)
    thresh = 0.4


    model.eval()
    with torch.no_grad():
        metric_sum = 0.0
        metric_count = 0
        for val_data in val_loader:
            val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                        )
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model,mode='gaussian')
            
            val_labels = val_labels.cpu().numpy()
            gt = np.squeeze(val_labels)
            val_outputs = act(val_outputs).cpu().numpy()
            seg= np.squeeze(val_outputs[0,1])
            seg[seg>thresh]=1
            seg[seg<thresh]=0
            value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))

            metric_count += 1
            metric_sum += value.sum().item()
        metric = metric_sum / metric_count
        print("Dice score:", metric)
                
          
#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)