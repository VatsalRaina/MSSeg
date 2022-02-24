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
parser.add_argument('--path_model', type=str, default='', help='Specify the dir to al the trained models')
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
    for frac in fracs_retained:
        pos = int(N * frac)
        if pos == N:
            curr_preds = preds
        else:
            curr_preds = np.concatenate((preds[:pos], gts[pos:]))
        dsc_scores.append(dice_metric(gts, curr_preds))
    dsc_scores = np.asarray(dsc_scores)

    if plot:

        plt.plot(fracs_retained, dsc_scores, label="Uncertainty")
        plt.xlabel("Retention Fraction")
        plt.ylabel("DSC")
        
        # Get ideal line
        uncs = np.absolute(gts-preds)
        ordering = uncs.argsort()
        uncs = uncs[ordering]
        gts = gts[ordering]
        preds = preds[ordering]
        dsc_scores_ideal = []
        for frac in fracs_retained:
            pos = int(N * frac)
            if pos == N:
                curr_preds = preds
            else:
                curr_preds = np.concatenate((preds[:pos], gts[pos:]))
            dsc_scores_ideal.append(dice_metric(gts, curr_preds))
        dsc_scores_ideal = np.asarray(dsc_scores_ideal)
        plt.plot(fracs_retained, dsc_scores_ideal, label="Ideal")

        # Get random line
        uncs = np.linspace(0, 1000, len(gts))
        np.random.seed(0)
        np.random.shuffle(uncs)
        ordering = uncs.argsort()
        uncs = uncs[ordering]
        gts = gts[ordering]
        preds = preds[ordering]
        dsc_scores_random = []
        for frac in fracs_retained:
            pos = int(N * frac)
            if pos == N:
                curr_preds = preds
            else:
                curr_preds = np.concatenate((preds[:pos], gts[pos:]))
            dsc_scores_random.append(dice_metric(gts, curr_preds))
        dsc_scores_random = np.asarray(dsc_scores_random)
        plt.plot(fracs_retained, dsc_scores_random, label="Random")

        plt.xlim([0.0,1.01])
        plt.legend()
        plt.savefig('unc_ret.png')
        plt.clf()

    return metrics.auc(fracs_retained, dsc_scores)
    

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

    all_vals = {
        "confidence": 0.0,
        "entropy_of_expected": 0.0,
        "expected_entropy": 0.0,
        "mutual_information": 0.0,
        "epkl": 0.0,
        "reverse_mutual_information": 0.0,
        "ideal": 0.0,
        "random": 0.0
        }
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
            uncs = ensemble_uncertainties_classification( np.concatenate( (np.expand_dims(all_outputs, axis=-1), np.expand_dims(1.-all_outputs, axis=-1)), axis=-1) )
            
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
            for unc_key, curr_uncs in uncs.items():
                auc_dsc = get_unc_score(gt.flatten(), seg.flatten(), curr_uncs.flatten(), plot=False)
                print(unc_key, auc_dsc)
                all_vals[unc_key] += 1. - auc_dsc

            # Get ideal values
            auc_dsc = get_unc_score(gt.flatten(), seg.flatten(), np.absolute(gt.flatten()-seg.flatten()), plot=False)
            print("Ideal", auc_dsc)
            all_vals["ideal"] += 1. - auc_dsc

            # Get random values
            rand_uncs = np.linspace(0, 1000, len(gt.flatten()))
            np.random.seed(0)
            np.random.shuffle(rand_uncs)
            auc_dsc = get_unc_score(gt.flatten(), seg.flatten(), rand_uncs, plot=False)
            print("Random", auc_dsc)
            all_vals["random"] += 1. - auc_dsc


            im_sum = np.sum(seg) + np.sum(gt)
            if im_sum == 0:
                value = 1.0
                dsc = value
            else:
                value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                dsc = value.sum().item()
            print("Dice score:", dsc)

    print("Mean across all patients:")
    
    for unc_key, sum_aacdsc in all_vals.items():
        print(unc_key, sum_aacdsc/num_patients)


    # # Plot the first ground truth and corresponding prediction at a random slice
    # gt, pred, unc = all_groundTruths[0], all_predictions[0], all_uncs[0]
    # gt_slice, pred_slice, unc_slice = gt[100,:,:], pred[100,:,:], unc[100,:,:]

    # sns.heatmap(gt_slice)
    # plt.savefig('gt.png')
    # plt.clf()

    # sns.heatmap(pred_slice)
    # plt.savefig('pred.png')
    # plt.clf()

    # sns.heatmap(unc_slice)
    # plt.savefig('unc.png')
    # plt.clf()

#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)