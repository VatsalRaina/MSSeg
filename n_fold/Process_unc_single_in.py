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
parser.add_argument('--path_data1', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts1', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_data2', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts2', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_data3', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts3', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_data4', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts4', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_model', type=str, default='', help='Specify the dir to al the trained models')


# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def dice_norm_metric(ground_truth, predictions):
    """
    For a single example returns DSC_norm, fpr, fnr
    """

    # Reference for normalized DSC
    r = 0.001
    # Cast to float32 type
    gt = ground_truth.astype("float32")
    seg = predictions.astype("float32")
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0, 1.0, 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1-r) * np.sum(gt) / ( r * ( len(gt.flatten()) - np.sum(gt) ) )
        tp = np.sum(seg[gt==1])
        fp = np.sum(seg[gt==0])
        fn = np.sum(gt[seg==0])
        fp_scaled = k * fp
        dsc_norm = 2 * tp / (fp_scaled + 2 * tp + fn)
        return dsc_norm


def get_unc_score(gts, preds, uncs, plot=True):
    ordering = uncs.argsort()
    uncs = uncs[ordering]
    gts = gts[ordering]
    preds = preds[ordering]
    N = len(gts)
    
    # # Significant class imbalance means it is important to use logspacing between values
    # # so that it is more granular for the higher retention fractions
    num_values = 200
    fracs_retained = np.log(np.arange(num_values+1)[1:])
    fracs_retained = fracs_retained / np.amax(fracs_retained)
    dsc_norm_scores = []

    for frac in fracs_retained:
        pos = int(N * frac)
        if pos == N:
            curr_preds = preds
        else:
            curr_preds = np.concatenate((preds[:pos], gts[pos:]))
        dsc_norm = dice_norm_metric(gts, curr_preds)
        dsc_norm_scores.append(dsc_norm)
    dsc_norm_scores = np.asarray(dsc_norm_scores)

    return metrics.auc(fracs_retained, dsc_norm_scores)
    

def main(args):

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    root_dir= args.path_model  # Path where the trained model is saved

    flair1 = sorted(glob(os.path.join(args.path_data1, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs1 = sorted(glob(os.path.join(args.path_gts1, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))        
    flair2 = sorted(glob(os.path.join(args.path_data2, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs2 = sorted(glob(os.path.join(args.path_gts2, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))
    flair3 = sorted(glob(os.path.join(args.path_data3, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs3 = sorted(glob(os.path.join(args.path_gts3, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))
    flair4 = sorted(glob(os.path.join(args.path_data4, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs4 = sorted(glob(os.path.join(args.path_gts4, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))

    test_files=[]
    for j in range(len(flair1)):
        test_files = test_files + [{"image": fl, "label": seg} for fl, seg in zip(flair1[j:j+1], segs1[j:j+1])]

    for j in range(len(flair2)):
        test_files = test_files + [{"image": fl, "label": seg} for fl, seg in zip(flair2[j:j+1], segs2[j:j+1])]

    for j in range(len(flair3)):
        test_files = test_files + [{"image": fl, "label": seg} for fl, seg in zip(flair3[j:j+1], segs3[j:j+1])]

    for j in range(len(flair4)):
        test_files = test_files + [{"image": fl, "label": seg} for fl, seg in zip(flair4[j:j+1], segs4[j:j+1])]

    # print("Testing cases:", len(test_files))
    
    # print()
    # print("-------------------------------------------------------------------")
    # print("Welcome!")
    # print()
    # print('The MS lesion segmentation will be computed on the following files: ')
    # print()
    # print(test_files)
    # print()
    # print("-------------------------------------------------------------------")
    # print()
    # print('Loading the files and the trained network')
   
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



    model = UNet(dimensions=3,in_channels=1, out_channels=2,channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),num_res_units=0).to(device)
     

    act = Activations(softmax=True)

    model.load_state_dict(torch.load(root_dir + "/Best_model_finetuning.pth"))
    model.eval()

    # print()
    # print('Running the inference, please wait... ')
    
    th = args.threshold

    all_vals_dsc_norm = {
        "unc": 0.0
        }

    num_patients = 0

    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            num_patients += 1
            # if count != args.patient_num:
            #     continue
            # print("Patient num: ", count)
            inputs, gt  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                     batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
            outputs_o = (act(outputs))
            outputs = act(outputs).cpu().numpy()
            outputs = np.squeeze(outputs[0,1])

            # Get all uncertainties
            uncs = -1 * ( outputs*np.log(outputs + 1e-10) + (1.-outputs)*np.log(1.-outputs + 1e-10) )
            
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


            # Calculate all AUC-DSCs

            auc_dsc_norm = get_unc_score(gt.flatten(), seg.flatten(), uncs.flatten(), plot=False)
            all_vals_dsc_norm['unc'] += 1. - auc_dsc_norm


    # print("Mean across all patients:")

    for unc_key, sum_aac in all_vals_dsc_norm.items():
        print(unc_key, sum_aac/num_patients)


#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)