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
parser.add_argument('--path_model1', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--path_model2', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--path_model3', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--patient_num', type=int, default=0, help='Specify 1 patient to permit parallel processing')


# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def dice_metric(ground_truth, predictions):
    """
    Returns Dice coefficient for a single example.
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target, 
                     with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [W, H, D].
    Returns:
      Dice coefficient overlap (`float` in [0.0, 1.0])
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    ground_truth = ground_truth.astype("float32")
    predictions = predictions.astype("float32")

    # Calculate intersection and union of y_true and y_predict
    intersection = np.sum(predictions * ground_truth)
    union = np.sum(predictions) + np.sum(ground_truth)

    # Calcualte dice metric
    if intersection == 0.0 and union == 0.0:
      dice = 1.0
    else:
      dice = (2. * intersection) / (union)

    return dice

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

        fpr = fp / ( len(gt.flatten()) - np.sum(gt) )
        if np.sum(gt) == 0:
            fnr = 1.0
        else:
            fnr = fn / np.sum(gt)
        return dsc_norm, fpr, fnr


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
    dsc_scores = []
    dsc_norm_scores = []
    fpr_scores = []
    fnr_scores = []
    for frac in fracs_retained:
        pos = int(N * frac)
        if pos == N:
            curr_preds = preds
        else:
            curr_preds = np.concatenate((preds[:pos], gts[pos:]))
        dsc_scores.append(dice_metric(gts, curr_preds))
        dsc_norm, fpr, fnr = dice_norm_metric(gts, curr_preds)
        dsc_norm_scores.append(dsc_norm)
        fpr_scores.append(fpr)
        fnr_scores.append(fnr)
    dsc_scores = np.asarray(dsc_scores)
    dsc_norm_scores = np.asarray(dsc_norm_scores)
    fpr_scores = np.asarray(fpr_scores)
    fnr_scores = np.asarray(fnr_scores)

    return fracs_retained, dsc_scores, dsc_norm_scores


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


    N = (len(flair)) 

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

    models1 = []
    for i in range(K):
        models1.append(UNet(dimensions=3,in_channels=1, out_channels=2,channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),num_res_units=0).to(device))

    models2 = []
    for i in range(K):
        models2.append(UNet(dimensions=3,in_channels=1, out_channels=2,channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),num_res_units=0).to(device))

    models3 = []
    for i in range(K):
        models3.append(UNet(dimensions=3,in_channels=1, out_channels=2,channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),num_res_units=0).to(device))

    act = Activations(softmax=True)
    
    for i, model in enumerate(models1):
        model.load_state_dict(torch.load(args.path_model1 + "seed" + str(i+1) + "/Best_model_finetuning.pth"))
        model.eval()

    for i, model in enumerate(models2):
        model.load_state_dict(torch.load(args.path_model2 + "seed" + str(i+1) + "/Best_model_finetuning.pth"))
        model.eval()

    for i, model in enumerate(models3):
        model.load_state_dict(torch.load(args.path_model3 + "seed" + str(i+1) + "/Best_model_finetuning.pth"))
        model.eval()

    print()
    print('Running the inference, please wait... ')
    
    th = args.threshold

    num_patients = 0

    all_curves_dsc1 = []
    all_curves_dsc_norm1 = []

    all_curves_dsc2 = []
    all_curves_dsc_norm2 = []

    all_curves_dsc3 = []
    all_curves_dsc_norm3 = []

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

            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)

            all_outputs = []
            for model in models1:
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
                outputs_o = (act(outputs))
                outputs = act(outputs).cpu().numpy()
                outputs = np.squeeze(outputs[0,1])
                all_outputs.append(outputs)
            all_outputs = np.asarray(all_outputs)
            outputs = np.mean(all_outputs, axis=0)

            # Get all uncertainties
            uncs = ensemble_uncertainties_classification( np.concatenate( (np.expand_dims(all_outputs, axis=-1), np.expand_dims(1.-all_outputs, axis=-1)), axis=-1) )
            
            outputs[outputs>th]=1
            outputs[outputs<th]=0
            seg= np.squeeze(outputs)

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
            for unc_key, curr_uncs in uncs.items():
                if unc_key != "reverse_mutual_information":
                    continue
                fracs_retained, dsc_curve, dsc_norm_curve = get_unc_score(gt.flatten(), seg.flatten(), curr_uncs.flatten())
            all_curves_dsc1.append(dsc_curve)
            all_curves_dsc_norm1.append(dsc_norm_curve)


            all_outputs = []
            for model in models2:
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
                outputs_o = (act(outputs))
                outputs = act(outputs).cpu().numpy()
                outputs = np.squeeze(outputs[0,1])
                all_outputs.append(outputs)
            all_outputs = np.asarray(all_outputs)
            outputs = np.mean(all_outputs, axis=0)

            # Get all uncertainties
            uncs = ensemble_uncertainties_classification( np.concatenate( (np.expand_dims(all_outputs, axis=-1), np.expand_dims(1.-all_outputs, axis=-1)), axis=-1) )
            
            outputs[outputs>th]=1
            outputs[outputs<th]=0
            seg= np.squeeze(outputs)

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
            for unc_key, curr_uncs in uncs.items():
                if unc_key != "reverse_mutual_information":
                    continue
                fracs_retained, dsc_curve, dsc_norm_curve = get_unc_score(gt.flatten(), seg.flatten(), curr_uncs.flatten())
            all_curves_dsc2.append(dsc_curve)
            all_curves_dsc_norm2.append(dsc_norm_curve)


            all_outputs = []
            for model in models3:
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
                outputs_o = (act(outputs))
                outputs = act(outputs).cpu().numpy()
                outputs = np.squeeze(outputs[0,1])
                all_outputs.append(outputs)
            all_outputs = np.asarray(all_outputs)
            outputs = np.mean(all_outputs, axis=0)

            # Get all uncertainties
            uncs = ensemble_uncertainties_classification( np.concatenate( (np.expand_dims(all_outputs, axis=-1), np.expand_dims(1.-all_outputs, axis=-1)), axis=-1) )
            
            outputs[outputs>th]=1
            outputs[outputs<th]=0
            seg= np.squeeze(outputs)

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
            for unc_key, curr_uncs in uncs.items():
                if unc_key != "reverse_mutual_information":
                    continue
                fracs_retained, dsc_curve, dsc_norm_curve = get_unc_score(gt.flatten(), seg.flatten(), curr_uncs.flatten())
            all_curves_dsc3.append(dsc_curve)
            all_curves_dsc_norm3.append(dsc_norm_curve)


    all_curves_dsc1 = np.asarray(all_curves_dsc1)
    all_curves_dsc_norm1 = np.asarray(all_curves_dsc_norm1)
    all_curves_dsc2 = np.asarray(all_curves_dsc2)
    all_curves_dsc_norm2 = np.asarray(all_curves_dsc_norm2)
    all_curves_dsc3 = np.asarray(all_curves_dsc3)
    all_curves_dsc_norm3 = np.asarray(all_curves_dsc_norm3)
    
    dsc_scores1 = np.mean(all_curves_dsc1, axis=0)
    dsc_norm_scores1 = np.mean(all_curves_dsc_norm1, axis=0)
    dsc_scores2 = np.mean(all_curves_dsc2, axis=0)
    dsc_norm_scores2 = np.mean(all_curves_dsc_norm2, axis=0)
    dsc_scores3 = np.mean(all_curves_dsc3, axis=0)
    dsc_norm_scores3 = np.mean(all_curves_dsc_norm3, axis=0)
    
    plt.plot(fracs_retained, dsc_scores1, label="MSSEG")
    plt.plot(fracs_retained, dsc_scores2, label="PubMRI")
    plt.plot(fracs_retained, dsc_scores3, label="Combined")
    plt.xlabel("Retention Fraction")
    plt.ylabel("DSC")
    plt.xlim([0.0,1.01])
    plt.legend()
    plt.savefig('unc_ret_dsc.png')
    plt.clf()

    plt.plot(fracs_retained, dsc_norm_scores1, label="MSSEG")
    plt.plot(fracs_retained, dsc_norm_scores2, label="PubMRI")
    plt.plot(fracs_retained, dsc_norm_scores3, label="Combined")
    plt.xlabel("Retention Fraction")
    plt.ylabel(r"$\overline{DSC}$")
    plt.xlim([0.0,1.01])
    plt.legend()
    plt.savefig('unc_ret_dsc_norm.png')
    plt.clf()

#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)