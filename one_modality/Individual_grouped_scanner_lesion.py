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
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for lesion detection')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_data2', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts2', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_save', type=str, default='', help='Specify the path to save the segmentations')

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
    segs = sorted(glob(os.path.join(args.path_gts, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))        


    N = (len(flair)) # Number of subjects for training/validation, by default using all subjects in the folder
    
    # np.random.seed(111)
    # indices = np.random.permutation(N)
    indices = np.arange(N)
    v=indices[:]

    test_files=[]
    for j in v:
        test_files = test_files + [{"image": fl, "label": seg} for fl, seg in zip(flair[j:j+1], segs[j:j+1])]

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
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image","label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "label"]),
    ]
    )
    #%%

    val_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)


    act = Activations(softmax=True)

    print()
    print('Running the inference, please wait... ')
    
    th = args.threshold

    verio_patients = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    discovery_patients = [11, 12, 13, 14, 15, 16, 17, 18]
    aera_patients = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    ingenia_patients = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

    verio_load = []
    discovery_load = []
    aera_load = []
    ingenia_load = []


    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            patient_num = count + 1

            inputs, gt  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                     batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)
  
            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)

            load = np.sum(gt) / len(gt.flatten())
            
            if patient_num in verio_patients:
                verio_load.append(load)
            if patient_num in discovery_patients:
                discovery_load.append(load)
            if patient_num in aera_patients:
                aera_load.append(load)
            if patient_num in ingenia_patients:
                ingenia_load.append(load)



    path_data = args.path_data2  # Path where the data is
    flair = sorted(glob(os.path.join(path_data, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(args.path_gts2, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))        


    N = (len(flair)) # Number of subjects for training/validation, by default using all subjects in the folder
    
    indices = np.arange(N)
    v=indices[:]

    test_files=[]
    for j in v:
        test_files = test_files + [{"image": fl, "label": seg} for fl, seg in zip(flair[j:j+1], segs[j:j+1])]

    print("Testing cases:", len(test_files))

    val_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    magnetomTrio_load = []
    magnetomTrio_load = []

    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            patient_num = count + 1

            inputs, gt  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                     batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)

            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)

            load = np.sum(gt) / len(gt.flatten())

            magnetomTrio_load.append(load)

    data = []

    for l in verio_load:
        data.append({"Scanner": "Verio", "Lesion load": l})
    for dsc in discovery_load:
        data.append({"Scanner": "Discovery", "Lesion load": l})
    for dsc in aera_load:
        data.append({"Scanner": "Aera", "Lesion load": l})
    for dsc in ingenia_load:
        data.append({"Scanner": "Ingenia", "Lesion load": l})
    for dsc in magnetomTrio_load:
        data.append({"Scanner": "Magnetom", "Lesion load": l})


    df = pd.DataFrame(data)
    
    ax = sns.boxplot(x="Scanner", y=r"Lesion load", data=df)
    #ax = sns.swarmplot(x="Scanner", y=r"$\overline{DSC}$", hue="Model", data=df, color=".25")
    plt.savefig('scanner_load.png')
    plt.clf()


#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)