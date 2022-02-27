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
    AddChanneld,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,ToTensord,ConcatItemsd,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, AsDiscrete, Activations)
from monai.utils import first, set_determinism
from monai.data import write_nifti, create_file_basename, NiftiDataset
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn import metrics

from Uncertainty import ensemble_uncertainties_classification

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for lesion detection')
parser.add_argument('--num_models', type=int, default=5, help='Number of models in ensemble')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--num_annotators', type=int, default=3, help='The number of annotators')
parser.add_argument('--path_model', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--patient_num', type=int, default=0, help='Specify 1 patient to permit parallel processing')



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

    root_dir= args.path_model  # Path where the trained model is saved
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

    K = args.num_models

    models = []
    for i in range(K):
        models.append(UNet(dimensions=3,in_channels=1, out_channels=2,channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),num_res_units=0).to(device))
     

    act = Activations(softmax=True)
    
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(root_dir + "seed" + str(i+1) + "/Best_model_finetuning.pth"))
        model.eval()

    print()
    print('Running the inference, please wait... ')
    
    th = args.threshold

    num_patients = 0

    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            num_patients += 1
            # if count != args.patient_num:
            #     continue
            print("Patient num: ", count)
            inputs, gt  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                     batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            # print(gt.cpu().numpy().shape)

            all_outputs = []
            for model in models:
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
                outputs_o = (act(outputs))
                outputs = act(outputs).cpu().numpy()
                outputs = np.squeeze(outputs[0,1])
                all_outputs.append(outputs)
            all_outputs = np.asarray(all_outputs)
            outputs = np.mean(all_outputs, axis=0)

            # Get all uncertainties
            uncs = ensemble_uncertainties_classification( np.concatenate( (np.expand_dims(all_outputs, axis=-1), np.expand_dims(1.-all_outputs, axis=-1)), axis=-1) )["reverse_mutual_information"]

            break

    # Get annotator variance at each voxel position

    annotator_paths = []
    for annotator_num in range(1, args.num_annotators+1):
        path = args.path_gts + "annotator" + str(annotator_num) + "/1_ann" + str(annotator_num) + ".nii.gz"
        annotator_paths.append(path)

    indices = np.arange(args.num_annotators)
    v=indices[:]

    ann_files = []
    for j in v:
        ann_files = ann_files + [{"image": flair[0], "annotator": seg} for seg in annotator_paths[j:j+1]]

    val_transforms = Compose(
    [
        LoadNiftid(keys=["image", "annotator"]),
        AddChanneld(keys=["image", "annotator"]),
        Spacingd(keys=["image", "annotator"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "annotator"]),
    ]
    )

    val_ds = CacheDataset(data=ann_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    all_annotations = []
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            num_patients += 1
            annotations  = (batch_data["annotator"].to(device).cpu().numpy())
            all_annotations.append(annotations[0][0].astype(float))
    
    all_annotations = np.asarray(all_annotations)
    # Assume each voxel position is a bernoulli distribution
    variance_map = np.mean(all_annotations, axis=0) * (1. - np.mean(all_annotations, axis=0))

    # Plot voxels of model uncertainty against annotator variance

    print(variance_map.shape)
    print(uncs.shape)

    sns.regplot(x=variance_map.flatten(), y=uncs[:,:256,:].flatten())
    plt.xlabel("Annotator variance")
    plt.ylabel("Predictive uncertainty")
    plt.savefig('correlation.png')
    plt.clf()


    # # Plot the first ground truth and corresponding prediction at a random slice
    # var_slice, unc_slice = variance_map[100,:,:], uncs[100,:,:]

    # sns.heatmap(unc_slice)
    # plt.savefig('unc.png')
    # plt.clf()

    # sns.heatmap(var_slice)
    # plt.savefig('var.png')
    # plt.clf()





#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)