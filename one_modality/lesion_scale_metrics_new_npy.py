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
from joblib import Parallel
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
from utils.retention_curve import lesions_uncertainty
from utils.visualise import plot_iqr_median_rc, plot_mean_rc
from utils.transforms import get_TP_FP_lesions_mask_parallel, get_FN_lesions_count_parallel

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
parser.add_argument('--IoU_threshold', default=0.5, type=float, help="IoU threshold for lesion F1 computation")

        
def main(args):
    npy_files = Path(args.path_data).glob("*data.npz")
        
    path_save = args.path_save
    path_les = os.path.join(path_save, "lesions_data")
    os.makedirs(args.path_save, exist_ok=True)
    os.makedirs(path_les, exist_ok=True)
    thresh_str = "%.3d" % (args.IoU_threshold * 100)
    
    # %%
    print("Staring evaluation")
    num_patients = 0
    fracs_ret = np.log(np.arange(200 + 1)[1:])
    fracs_ret /= np.amax(fracs_ret)
    uncs_metrics = ['sum', 'mean', 'logsum', 'volume']
    uncs_metrics = uncs_metrics + [um + '_ext' for um in uncs_metrics] + ['mean_iou_det', 'iou_ap_det']
    uncs_metrics_ = uncs_metrics + ['hash', 'type', 'number']
    
    les_uncs_df = pd.DataFrame([], columns=uncs_metrics_)
    csv_path = os.path.join(path_save, f'lesion_uncertainty_metrics_{args.uncs_metric}.csv')
    n_fn_df = pd.DataFrame([], columns=['fn_count', 'hash'])
    csv1_path = os.path.join(path_save, 'fn_count.csv')
    
    with Parallel(n_jobs=args.n_jobs) as parallel_backend:
        for npy_file in npy_files:
            # loading variables
            npy_loader = np.load(npy_file)
            
            filename_or_obj = re.sub("_data.npy", "", os.path.basename(npy_file))
            
            all_outputs = npy_loader['all_outputs']
            gt=npy_loader['gt']
            seg=npy_loader['seg']
            uncs_mask=npy_loader['multi_uncs_mask']
            uncs_value=npy_loader['uncs_value']
            seg=npy_loader['seg']
                    
            ens_seg = all_outputs.copy()
            ens_seg[ens_seg >= args.threshold] = 1.0
            ens_seg[ens_seg < args.threshold] = 0.0
            
            del all_outputs
    
            # computing uncs measures
            n_fn = get_FN_lesions_count_parallel(ground_truth=gt,
                                               predictions=seg,
                                               IoU_threshold=args.IoU_threshold,
                                               mask_type='binary',
                                               parallel_backend=parallel_backend)
            
            tp_lesions, fp_lesions = get_TP_FP_lesions_mask_parallel(ground_truth=gt,
                                                      predictions=seg,
                                                      IoU_threshold=args.IoU_threshold,
                                                      mask_type='binary',
                                                      parallel_backend=parallel_backend)
            
            np.savez_compressed(os.path.join(path_les, filename_or_obj + 'lesion_data.npz'), 
                                tp_lesions=tp_lesions, fp_lesions=fp_lesions, n_fn=n_fn)
            # get list of lesions metrics
            uncs_list_tp = lesions_uncertainty(uncs_map=uncs_value, 
                                                binary_mask=tp_lesions, 
                                                uncs_multi_mask=uncs_mask, 
                                                ens_pred=ens_seg, 
                                                parallel=parallel_backend,
                                                dl=False)
            uncs_list_fp = lesions_uncertainty(uncs_map=uncs_value, 
                                                binary_mask=fp_lesions, 
                                                uncs_multi_mask=uncs_mask, 
                                                ens_pred=ens_seg, 
                                                parallel=parallel_backend,
                                                dl=False)
            
            # add lesions descriptions to the lesions dictioaries
            for i_l, les_dict in enumerate(uncs_list_tp):
                les_dict['type'] = 'tp'
                les_dict['number'] = i_l
                les_dict['hash'] = filename_or_obj
                les_uncs_df = les_uncs_df.append(les_dict, ignore_index=True)
                
            for i_l, les_dict in enumerate(uncs_list_fp):
                les_dict['type'] = 'tp'
                les_dict['number'] = i_l
                les_dict['hash'] = filename_or_obj
                les_uncs_df = les_uncs_df.append(les_dict, ignore_index=True)
                
            n_fn_df = n_fn_df.append({'fn_count': n_fn, 'hash': filename_or_obj})
                
            les_uncs_df.to_csv(csv_path)
            n_fn_df.to_csv(csv1_path)
   
    # get an optimal threshold
    print(f"Saved resutls to folder {args.path_save}")

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
