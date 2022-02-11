"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
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
    AddChanneld,AddChannel,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,Spacing,ToTensord,ConcatItemsd,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, AsDiscrete, Activations)
from monai.utils import first, set_determinism
from monai.data import write_nifti, create_file_basename, NiftiDataset
import numpy as np
from scipy import ndimage
import itertools

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--path_gt', type=str, default='', help='Path to ground truths')
parser.add_argument('--path_pred', type=str, default='', help='Path to segmentation prediction')



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

    segs = sorted(glob(os.path.join(args.path_pred, "*.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    gts = sorted(glob(os.path.join(args.path_gt, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))  


    N = (len(segs)) 
    
    # np.random.seed(111)
    # indices = np.random.permutation(N)
    indices = np.arange(N)
    v=indices[:]

    test_files=[]
    for j in v:
        test_files = test_files + [{"seg": seg, "gt": gt} for seg, gt in zip(segs[j:j+1], gts[j:j+1])]

    print("Testing cases:", len(test_files))
    
    print()
    print("-------------------------------------------------------------------")
    print("Welcome!")
    print()
    print('The MS lesion segmentation will be computed on the following files: ')
    print()
    print(test_files)
    print()
    print("-------------------------------------------------------------------")
    print()
    print('Loading the files and the trained network')
   
    val_transforms = Compose(
    [
        LoadNiftid(keys=["seg", "gt"]),
        AddChanneld(keys=["seg","gt"]),
        ToTensord(keys=["seg", "gt"]),
    ]
    )
    #%%

    val_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    print()
    print('Running the inference, please wait... ')
    
    metric_sum = 0.0
    metric_count = 0
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            seg, gt = (
                    batch_data["seg"].to(device),
                    batch_data["gt"].type(torch.LongTensor).to(device),)

            seg = seg.cpu().numpy()
            seg = np.squeeze(seg)

            gt = gt.cpu().numpy()
            gt = np.squeeze(gt)

            im_sum = np.sum(seg) + np.sum(gt)
            if im_sum == 0:
                value = 1.0
                metric_sum += value
            else:
                value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                metric_sum += value.sum().item()
            metric_count += 1

        metric = metric_sum / metric_count
        print("Dice score:", metric)
            
    
#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)