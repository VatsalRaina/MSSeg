"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
         Adapted by Nataliia Molchanova
         
lesion_scale_retention_curve.py \
--threshold 0.35 \
--num_models 5 \
--path_data /home/meri/data/canonical/dev_in \
--path_gts /home/meri/data/canonical/dev_in \
--path_model /home/meri/uncertainty_challenge/models_cv/ex_ljubljana/ \
--path_save /home/meri/uncertainty_challenge/results/f1_lesion_score/ \
--flair_prefix FLAIR.nii.gz \
--gts_prefix gt.nii \
--check_dataset \
--n_jobs 10 --num_workers 10 \
--IoU_threshold 0.5

         
runfile('/home/meri/Code/MSSeg/one_modality/lesion_scale_retention_curve.py', wdir='/home/meri/Code/MSSeg/one_modality', args='--threshold 0.35 --num_models 5 --path_data /home/meri/data/canonical/dev_in --path_gts /home/meri/data/canonical/dev_in --path_model /home/meri/uncertainty_challenge/models_cv/ex_ljubljana/ --path_save /home/meri/uncertainty_challenge/results/f1_lesion_score/ --flair_prefix FLAIR.nii.gz --gts_prefix gt.nii --check_dataset --n_jobs 10 --num_workers 10 --IoU_threshold 0.25')
"""

import argparse
import os
from functools import partial
import numpy as np

import torch
import re
from glob import glob

from joblib import Parallel, delayed
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld, Compose, CropForegroundd, LoadNiftid, Orientationd, RandCropByPosNegLabeld,
    ScaleIntensityRanged, Spacingd, ToTensord, ConcatItemsd, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd, AsDiscrete, Activations)
import seaborn as sns;
import matplotlib.pyplot as plt
from sklearn import metrics
import json

sns.set_theme()
from Uncertainty import ensemble_uncertainties_classification, lesions_uncertainty_sum
import pandas as pd
from utils.data_load import *
from utils.setup import get_default_device
from utils.metrics import get_metric_for_rc_lesion
from utils.visualise import plot_iqr_median_rc, plot_mean_rc

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for lesion detection')
parser.add_argument('--patient_num', type=int, default=0, help='Specify 1 patient to permit parallel processing')
parser.add_argument('--num_models', type=int, default=5, help='Number of models in ensemble')
parser.add_argument('--i_seed', type=int, default=-1,
                    help='if num_models==1, it will use the i_seed model for predictions')
parser.add_argument('--unc_metric', type=str, default="reverse_mutual_information",
                    help='name of a single uncertainty metric')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_model', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--path_save', type=str, default='', help='Specify the path to save the segmentations')
parser.add_argument('--flair_prefix', type=str, default="FLAIR.nii.gz", help='name ending FLAIR')
parser.add_argument('--gts_prefix', type=str, default="gt.nii", help='name ending segmentation mask')
parser.add_argument('--check_dataset', action='store_true',
                    help='if sent, checks that FLAIR and semg masks names correspond to each other')
parser.add_argument('--n_jobs', type=int, default=1,
                    help="number of cores used for computation of dsc for different retention fractions")
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of jobs used to load the data, DataLoader parameter')
parser.add_argument('--IoU_threshold', default=0.25, type=float, help="IoU threshold for lesion F1 computation")
parser.add_argument('--lesion_unc_type', type=str, default='average', help='Approach for calculating lesion uncertainty')



def main(args):
    device = get_default_device()

    val_loader = get_data_loader(args)

    # %%
    print("Initializing the models")

    n_models = args.num_models

    models = []
    for i in range(n_models):
        models.append(UNet(dimensions=3, in_channels=1, out_channels=2, channels=(32, 64, 128, 256, 512),
                           strides=(2, 2, 2, 2), num_res_units=0).to(device))

    act = Activations(softmax=True)

    if n_models > 1:
        for i, model in enumerate(models):
            pthfilepath = os.path.join(args.path_model, "seed" + str(i + 1), "Best_model_finetuning.pth")
            model.load_state_dict(torch.load(pthfilepath))
            model.eval()
    elif n_models == 1:
        pthfilepath = os.path.join(args.path_model, "seed" + str(args.i_seed), "Best_model_finetuning.pth")
        models[0].load_state_dict(torch.load(pthfilepath))
        models[0].eval()
    else:
        raise ValueError(f"invalid number of num_models {args.num_models}")
        
    os.makedirs(args.path_save, exist_ok=True)
    thresh_str = "%.3d" % (args.IoU_threshold * 100)
    
    # %%
    print("Staring evaluation")
    num_patients = 0
    fracs_ret = np.log(np.arange(200 + 1)[1:])
    fracs_ret /= np.amax(fracs_ret)
    #unc_types = ['average', 'sum', 'count']
    unc_types = ['average', 'sum']
    metric_rf_df_1 = pd.DataFrame([], columns=fracs_ret)
 
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            if count==2:
                break
            # Get models predictions
            num_patients += 1
            inputs, gt = (
                batch_data["image"].to(device),
                batch_data["label"].type(torch.LongTensor).to(device),
            )
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            all_outputs = []
            for model in models:
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
                outputs = act(outputs).cpu().numpy()        # [1, 2, H, W, D]
                all_outputs.append(np.squeeze(outputs[0, 1]))
            all_outputs = np.asarray(all_outputs)        # [M, H, W, D]
            outputs_mean = np.mean(all_outputs, axis=0)     # [H, W, D]

            outputs_mean[outputs_mean > args.threshold] = 1
            outputs_mean[outputs_mean < args.threshold] = 0
            seg = np.squeeze(outputs_mean)      # [H, W, D]
            seg = remove_connected_components(segmentation=seg, l_min=9)
            seg[seg >= args.threshold] = 1.
            seg[seg < args.threshold] = 0.

            val_labels = gt.cpu().numpy()   # [1, 1, H, W, D]
            gt = np.squeeze(val_labels)     # [H, W, D]

            # Get all uncertainties
            uncs_value = ensemble_uncertainties_classification(
                np.concatenate((np.expand_dims(all_outputs, axis=-1),
                                np.expand_dims(1. - all_outputs, axis=-1)),
                               axis=-1))[args.unc_metric]   # [H, W, D]



            metric_rf, f1_values = get_metric_for_rc_lesion(gts=gt,
                                                preds=seg,
                                                uncs=uncs_value,
                                                fracs_retained=fracs_ret,
                                                IoU_threshold=args.IoU_threshold,
                                                n_jobs=args.n_jobs,
                                                unc_type=u'average')
            row_df = pd.DataFrame(np.expand_dims(metric_rf, axis=0), 
                                columns=fracs_ret, index=[0])
            metric_rf_df_1.append(row_df, ignore_index=True)
            
            if num_patients % 1 == 0:
                print(f"Processed {num_patients} scans")

    # mean_aac = 1. - metrics.auc(fracs_ret, np.asarray(metric_rf_df.mean()))
    
    # with open(os.path.join(args.path_save, f"f1{thresh_str}-AAC_{args.unc_metric}.csv"), 'w') as f:
    #     f.write(f"mean AAC:\t{mean_aac}")
    

    mean_average = metric_rf_df_1.mean()
    # print(mean_average)
    plt.plot(fracs_ret, mean_average, label='Average')

    plt.xlim([0, 1.01])
    plt.legend()
    plt.savefig('./ret_all.png')
    
    print(f"Saved f1 scores and retention curve to folder {args.path_save}")

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
