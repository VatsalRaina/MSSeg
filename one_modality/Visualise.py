"""
Visualise the 100th slice in the correct orientation for all patients in the provided test set
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
import matplotlib.cm as cm
import seaborn as sns; sns.set_theme()

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_unprocessed_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_save', type=str, default='', help='Specify the path to save the images along a given axis')

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

    path_data = args.path_data  # Path where the data is
    flair = sorted(glob(os.path.join(path_data, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    t1 = sorted(glob(os.path.join(path_data, "*T1.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))
    segs = sorted(glob(os.path.join(args.path_gts, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))        

    flair_un = sorted(glob(os.path.join(args.path_unprocessed_data, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    t1_un = sorted(glob(os.path.join(args.path_unprocessed_data, "*T1.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))

    N = (len(flair)) # Number of subjects for training/validation, by default using all subjects in the folder
    
    # np.random.seed(111)
    # indices = np.random.permutation(N)
    indices = np.arange(N)
    v=indices[:]

    test_files=[]
    for j in v:
        test_files = test_files + [{"flair": fl, "t1":t, "flair_un":fl_un, "label": seg} for fl, t, fl_un, seg in zip(flair[j:j+1], t1[j:j+1], flair_un[j:j+1], segs[j:j+1])]

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
        LoadNiftid(keys=["flair", "t1", "flair_un", "label"]),
        AddChanneld(keys=["flair", "t1", "flair_un", "label"]),
        Spacingd(keys=["flair", "t1", "flair_un", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "bilinear", "nearest")),
        NormalizeIntensityd(keys=["flair", "t1", "flair_un"], nonzero=True),
        ConcatItemsd(keys=["flair", "t1", "flair_un"], name="image"),
        ToTensord(keys=["image", "label"]),
    ]
    )
    #%%

    val_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
         
    print()
    print('Running the inference, please wait... ')
    
    all_groundTruths = []
    all_flair = []
    all_t1 = []
    all_flair_un = []
    all_t1_un = []

    with torch.no_grad():
        metric_sum = 0.0
        metric_count = 0
        for count, batch_data in enumerate(val_loader):
            inputs, gt  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                     batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)

            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)

            all_groundTruths.append(gt)
            all_flair.append(inputs.cpu().numpy()[0][0])
            all_t1.append(inputs.cpu().numpy()[0][1])
            all_flair_un.append(inputs.cpu().numpy()[0][2])
            break

    patient_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for patient_num in patient_nums:        
        gt, fl, t1, fl_un = all_groundTruths[patient_num-1], all_flair[patient_num-1], all_t1[patient_num-1], all_flair_un[patient_num-1]

        slice_num = 100
        gt_slice, fl_slice, t1_slice, fl_un_slice= np.rot90(gt[slice_num,:,:], 3), np.rot90(fl[slice_num,:,:], 3), np.rot90(t1[slice_num,:,:], 3), np.rot90(fl_un[slice_num,:,:], 3)

        ax = sns.heatmap(fl_slice, cbar=False, cmap="Blues", xticklabels=False, yticklabels=False, alpha=0.5)
        ax.invert_yaxis()
        plt.savefig(args.path_save + str(patient_num) + 'flair.png')
        # plt.clf()

        ax = sns.heatmap(t1_slice, cbar=False, cmap="Reds" xticklabels=False, yticklabels=False, alpha=0.5)
        ax.invert_yaxis()
        plt.savefig(args.path_save + str(patient_num) + 't1.png')
        plt.clf()

        # ax = sns.heatmap(fl_un_slice, cbar=False, cmap=cm.gray, xticklabels=False, yticklabels=False)
        # ax.invert_yaxis()
        # plt.savefig(args.path_save + str(patient_num) + 'flair_un.png')
        # plt.clf()

        # ax = sns.heatmap(gt_slice, cbar=False, cmap=cm.gray, xticklabels=False, yticklabels=False)
        # ax.invert_yaxis()
        # plt.savefig(args.path_save + str(patient_num) + 'gt.png')
        # plt.clf()

        break

#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)