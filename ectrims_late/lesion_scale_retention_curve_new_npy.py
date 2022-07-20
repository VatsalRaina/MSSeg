# -*- coding: utf-8 -*-

"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
         Adapted by Nataliia Molchanova
         
"""

import argparse
import os
from functools import partial
import numpy as np
from pathlib import Path
import torch
import re
from monai.data import write_nifti
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld, Compose, CropForegroundd, LoadNiftid, Orientationd, RandCropByPosNegLabeld,
    ScaleIntensityRanged, Spacingd, ToTensord, ConcatItemsd, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd, AsDiscrete, Activations)
import seaborn as sns;
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle

sns.set_theme()
from Uncertainty import ensemble_uncertainties_classification
import pandas as pd
from scipy import ndimage
from utils.data_load import remove_connected_components, get_data_loader
from utils.setup import get_default_device
from utils.retention_curve import get_lesion_rc_with_fn
from utils.visualise import plot_iqr_median_rc, plot_mean_rc

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.35, help='Threshold for lesion detection')

parser.add_argument('--num_models', type=int, default=5, help='Number of models in ensemble')
parser.add_argument('--i_seed', type=int, default=-1,
                    help='if num_models==1, it will use the i_seed model for predictions')
parser.add_argument('--unc_metric', type=str, default="reverse_mutual_information",
                    help='name of a single uncertainty metric')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_save', type=str, default='', help='Specify the path to save the segmentations')
parser.add_argument('--n_jobs', type=int, default=1,
                    help="number of cores used for computation of dsc for different retention fractions")
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of jobs used to load the data, DataLoader parameter')
parser.add_argument('--IoU_threshold', default=0.5, type=float, help="IoU threshold for lesion F1 computation")

def main(args):
    npy_files = Path(args.path_data).glob("*data.npz")
        
    os.makedirs(args.path_save, exist_ok=True)
    thresh_str = "%.3d" % (args.IoU_threshold * 100)
    
    # %%
    print("Staring evaluation")
    num_patients = 0
    fracs_ret = np.log(np.arange(200 + 1)[1:])
    fracs_ret /= np.amax(fracs_ret)
    uncs_metrics = ['sum', 'mean', 'logsum', 'volume']
    uncs_metrics = uncs_metrics + [um + '_ext' for um in uncs_metrics] + ['mean_iou_det', 'iou_ap_det']
    uncs_metrics_ = uncs_metrics + ['hash']
    
    metric_rf = dict(zip(
        uncs_metrics_, [pd.DataFrame([], columns=fracs_ret) for _ in uncs_metrics_]
    ))
    f1_dict = dict()
    
    for npy_file in npy_files:
        # loading variables
        npy_loader = np.load(npy_file)
        
        filename_or_obj = re.sub("_data", "", os.path.basename(npy_file))
        
        all_outputs = npy_loader['all_outputs']
        gt=npy_loader['gt']
        uncs_mask=npy_loader['multi_uncs_mask']
        uncs_value=npy_loader['uncs_value']
        seg=npy_loader['seg']
                
        ens_seg = all_outputs.copy()
        ens_seg[ens_seg >= args.threshold] = 1.0
        ens_seg[ens_seg < args.threshold] = 0.0
        
        del all_outputs

        # computing retention curves
        metric_rf_row, f1_values = get_lesion_rc_with_fn(gts=gt,
                                                 preds=seg,
                                                 uncs=uncs_value,
                                                 fracs_retained=fracs_ret,
                                                 multi_uncs_mask=uncs_mask,
                                                 IoU_threshold=args.IoU_threshold,
                                                 n_jobs=args.n_jobs,
                                                 all_outputs=ens_seg)
        for ut in uncs_metrics:
            row_df = pd.DataFrame(np.expand_dims(metric_rf_row[ut], axis=0), 
                                  columns=fracs_ret, index=[0])
            row_df['hash'] = filename_or_obj
            metric_rf[ut] = metric_rf[ut].append(row_df, ignore_index=True)
            
        f1_dict[filename_or_obj] = f1_values
        
        pkl_filepath = os.path.join(args.path_save, f"subject_f1{thresh_str}_{args.unc_metric}.pkl")
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(f1_dict, f)
            
        pkl_filepath = os.path.join(args.path_save, f"RC_f1{thresh_str}_{args.unc_metric}.pkl")
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(metric_rf, f)
        
        if num_patients % 10 == 0:
            print(f"Processed {num_patients} scans")
    
    # save auc
    mean_auc_ut = []
    with open(os.path.join(args.path_save, f"f1{thresh_str}-AUC_{args.unc_metric}.csv"), 'w') as f:
        for ut in uncs_metrics:
            mean_auc = metrics.auc(fracs_ret, np.asarray(metric_rf[ut].mean()))
            mean_auc_ut += [mean_auc]
            f.write(f"{ut} AAC:\t{mean_auc}")
            
    # plot rc
    save_path = os.path.join(args.path_save, 
                             f"mean_RC_f1{thresh_str}_{args.unc_metric}_df.png")
    for i_ut, ut in enumerate(uncs_metrics):
        mean = metric_rf[ut].mean()
        plt.plot(fracs_ret, mean, label=f"{ut} (R-AUC={mean_auc_ut[i_ut]:.3f})")

    plt.xlim([0, 1.01])
    plt.legend()
    plt.xlabel("Retention fraction")
    plt.ylabel("F1 lesion-scale")
    plt.savefig(save_path, dpi=300)
    plt.clf()
    
    print(f"Saved f1 scores and retention curve to folder {args.path_save}")

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
