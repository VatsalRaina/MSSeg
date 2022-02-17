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

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for lesion detection')
parser.add_argument('--num_models', type=int, default=5, help='Number of models in ensemble')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_model', type=str, default='', help='Specify the dir to all the trained models')
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

    root_dir= args.path_model  # Path where the trained model is saved
    path_data = args.path_data  # Path where the data is
    flair = sorted(glob(os.path.join(path_data, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(path_data, "*gt.nii")),
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

    
    all_predictions = []
    all_groundTruths = []
    all_uncs = []
    all_inputs = []

    with torch.no_grad():
        metric_sum = 0.0
        metric_count = 0
        for count, batch_data in enumerate(val_loader):
            inputs, gt  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                     batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            all_outputs = []
            for model in models:
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
                outputs_o = (act(outputs))
                outputs = act(outputs).cpu().numpy()
                outputs = np.squeeze(outputs[0,1])
                all_outputs.append(outputs)
            all_outputs = np.asarray(all_outputs)
            outputs = np.mean(all_outputs, axis=0)

            # Get all 
            uncs = -1 * (outputs * np.log(outputs) + (1. - outputs) * np.log(1. - outputs))
            
            outputs[outputs>th]=1
            outputs[outputs<th]=0
            seg= np.squeeze(outputs)
  
            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)

            """
            Remove connected components smaller than 10 voxels
            """
            l_min = 9
            labeled_seg, num_labels = ndimage.label(seg)
            label_list = np.unique(labeled_seg)
            num_elements_by_lesion = ndimage.labeled_comprehension(seg,labeled_seg,label_list,np.sum,float, 0)

            seg2 = np.zeros_like(seg)
            for l in range(len(num_elements_by_lesion)):
                if num_elements_by_lesion[l] > l_min:
            # assign voxels to output
                    current_voxels = np.stack(np.where(labeled_seg == l), axis=1)
                    seg2[current_voxels[:, 0],
                        current_voxels[:, 1],
                        current_voxels[:, 2]] = 1
            seg=np.copy(seg2) 

            all_predictions.append(seg)
            all_groundTruths.append(gt)
            all_uncs.append(uncs)
            all_inputs.append(inputs.cpu().numpy()[0])

            # im_sum = np.sum(seg) + np.sum(gt)
            # if im_sum == 0:
            #     value = 1.0
            #     metric_sum += value
            # else:
            #     value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
            #     metric_sum += value.sum().item()
            # metric_count += 1

        # metric = metric_sum / metric_count
        # print("Dice score:", metric)
            
    # Plot the first ground truth and corresponding prediction and uncertainty at a random slice
    gt, pred, unc, inp_f = all_groundTruths[0], all_predictions[0], all_uncs[0], all_inputs[0]

    for slice_num in range(len(gt)):

        gt_slice, pred_slice, unc_slice, inp_slice = gt[slice_num,:,:], pred[slice_num,:,:], unc[slice_num,:,:], inp_f[slice_num,:,:]

        # ax = sns.heatmap(gt_slice, vmin=0.0, vmax=1.0, cbar=False, xticklabels=False, yticklabels=False)
        # ax.invert_yaxis()
        # plt.savefig(args.path_save + str(slice_num) + 'gt.png')
        # plt.clf()

        # ax = sns.heatmap(pred_slice, vmin=0.0, vmax=1.0, cbar=False, xticklabels=False, yticklabels=False)
        # ax.invert_yaxis()
        # plt.savefig(args.path_save + str(slice_num) + 'pred.png')
        # plt.clf()

        # ax = sns.heatmap(unc_slice, vmin=0.0, vmax=np.log(2), cbar=False, xticklabels=False, yticklabels=False)
        # ax.invert_yaxis()
        # plt.savefig(args.path_save + str(slice_num) + 'unc.png')
        # plt.clf()

        ax = sns.heatmap(inp_slice, xticklabels=False, yticklabels=False)
        ax.invert_yaxis()
        plt.savefig(args.path_save + str(slice_num) + 'inp.png')
        plt.clf()

#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)