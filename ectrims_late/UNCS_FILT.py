#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:35:39 2022

@author: meri

Metrics to be obtained on the test set:
    * nDSC and F1 for CL and WM lesions separately
    * lesion uncertainty for predicted lesions (uncertainty = average mutual information)
    
    
--threshold 0.4 \
	--num_models 3 \
	--path_data "/home/meri/data/insider/dataset/test" \
	--path_gts "/home/meri/data/insider/dataset/test" \
	--path_model "/home/meri/ectrims/training" \
--root_brain_seg /home/meri/data/insider/freesurfer \
--path_gtsCL /home/meri/data/insider/gt_CL \
--path_gtsWM /home/meri/data/insider/gt_WM \
--path_save /home/meri/ectrims/evaluation \
--mask_prefix brain-labeled.nii.gz \
--gtsCL_prefix revised-202106_binary_cortical_lesions.nii.gz \
--gtsWM_prefix wm_lesions.nii.gz \
	--flair_prefix "FLAIR.nii.gz" \
	--gts_prefix "lesions.nii.gz" \
	--n_jobs 10 --num_workers 8
"""

import argparse
import os
from functools import partial
import numpy as np

import torch
import re
from glob import glob
import random

from joblib import Parallel, delayed
from monai.data import CacheDataset, DataLoader, Dataset, write_nifti
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld, Compose, CropForegroundd, LoadNiftid, Orientationd, RandCropByPosNegLabeld,
    ScaleIntensityRanged, Spacingd, ToTensord, ConcatItemsd, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd, AsDiscrete, Activations,
    Lambdad)
import seaborn as sns;
import matplotlib.pyplot as plt
import json
from pathlib import Path

sns.set_theme()
# from Uncertainty import ensemble_uncertainties_classification, lesions_uncertainty_sum
import pandas as pd
from scipy import ndimage
import logging
# from utils.data_load import *
# from utils.setup import get_default_device
# from utils.transforms import get_FN_lesions_mask_parallel, get_TP_FP_lesions_mask_parallel
# from utils.visualise import plot_iqr_median_rc, plot_mean_rc
import sys 
sys.path.insert(1, os.path.dirname(os.getcwd()))
from utils.data_load import remove_connected_components
from utils.metrics import dice_metric, lesion_metrics_parallel, lfpr, ltpr,tpr_fpr, check_binary_mask, intersection_over_union
from utils.setup import get_default_device

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.4, help='Threshold for lesion detection')
parser.add_argument('--lmin', type=int, default=6,
                    help="minimal size of predicted lesions")
parser.add_argument('--num_models', type=int, default=3, help='Number of models in ensemble')
parser.add_argument('--i_seed', type=int, default=-1,
                    help='if num_models==1, it will use the i_seed model for predictions')
parser.add_argument('--path_model', type=str, 
                    default='/home/meri/ectrims/cluster_training/wml_cl_detection_train', 
                    help='Specify the dir to al the trained models')
parser.add_argument('--path_save', type=str, default='/home/meri/ectrims/last_call/split2_eval', 
                    help='Specify the path to save the segmentations')
parser.add_argument('--n_jobs', type=int, default=10,
                    help="number of cores used for computation of dsc for different retention fractions")

def uncertainty_filtering(binary_mask, ens_pred, uncs_threshold, parallel):
    """
    Parallel evaluation of all uncertainties
    """
    def cc_uncs(cc_mask, ens_pred):
        ens_ious = []
        for m in range(ens_pred.shape[0]):
            multi_mask_ens = ndimage.label(ens_pred[m])[0]
            max_iou = 0.0
            for intersec_label in np.unique(cc_mask * multi_mask_ens):
                if intersec_label != 0.0:
                    iou = intersection_over_union(cc_mask, 
                                                  (multi_mask_ens == intersec_label).astype(float))
                    if iou > max_iou:
                        max_iou = iou
            ens_ious.append(max_iou)
        return 1 - np.mean(ens_ious)
    
    multi_mask, n_labels = ndimage.label(binary_mask)
    cc_labels = np.arange(1, n_labels + 1)
    process = partial(cc_uncs, ens_pred=ens_pred)
    les_uncs_list = parallel(delayed(process)(
        cc_mask=(multi_mask == cc_label).astype("float")
        ) for cc_label in cc_labels
    )
    # filter
    cleaned_mask = multi_mask.copy()
    for i_l, uncs in enumerate(les_uncs_list):
        if uncs > uncs_threshold:
            cleaned_mask -= (multi_mask == i_l + 1).astype("int32")
        
    return (cleaned_mask > 0).astype("float")

def evaluate(seg, gt, other_mask, parallel):
    check_binary_mask(seg)
    check_binary_mask(gt)
    if other_mask is not None:
        check_binary_mask(other_mask)
    
    dsc = dice_metric(gt, seg)
    
    f1, tpr_les_25, fnr_les_25, fpr_les_25, ppv_les_25  = lesion_metrics_parallel(ground_truth=gt, predictions=seg, 
                                                       IoU_threshold=0.25, 
                                                       other_mask=other_mask,
                                                       parallel_backend=parallel)
    lfpr_ = lfpr(seg=seg, gt=gt, other_mask=other_mask)
    ltpr_ = ltpr(seg=seg, gt=gt)
    
    fp = np.sum(seg[gt==0])
    tp = np.sum(seg[gt==1])
    
    fp_tn = np.sum(gt == 0)
    tp_fn = np.sum(gt == 1)
    fpr = fp / fp_tn if fp_tn > 0.0 else 0.0
    tpr = tp / tp_fn if tp_fn > 0.0 else 1.0
    
    ppv = tp / (tp + fp) if tp + fp > 0.0 else 1.0
    
    return {
        'dice': dsc, 'f1_les_25': f1, 
        'lfpr': lfpr_, 'ltpr': ltpr_,
        'fpr_les_25': fpr_les_25, 'tpr_les_25': tpr_les_25, 'fnr_les_25': fnr_les_25,
        'ppv': ppv, 'ppv_les_25': ppv_les_25,
        'tpr': tpr, 'fpr': fpr
    }

def check_dataset(filepaths_list, prefixes):
    """Check that there are equal amounts of files in both lists and
    the names before prefices are similar for each of the matched pairs of
    flair image filepath and gts filepath.
    Parameters:
        - flair_filepaths (list) - list of paths to flair images
        - gts_filepaths (list) - list of paths to lesion masks of flair images
        - args (argparse.ArgumentParser) - system arguments, should contain `flair_prefix` and `gts_prefix`
    """
    if len(filepaths_list) > 1:
        filepaths_ref = filepaths_list[0]
        for filepaths in filepaths_list[1:]:
            assert len(filepaths_ref) == len(
                filepaths), f"Found {len(filepaths_ref)} ref files and {len(filepaths)}"
        for sub_filepaths in list(map(list, zip(*filepaths_list))):
            filepath_ref = sub_filepaths[0]
            prefix_ref = prefixes[0]
            for filepath, prefix in zip(sub_filepaths[1:], prefixes[1:]):
                if os.path.basename(filepath_ref)[:-len(prefix_ref)] != os.path.basename(filepath)[:-len(prefix)]:
                    raise ValueError(f"{filepath_ref} and {filepath} do not match")


def get_data_loader_with_cl(path_flair, path_mp2rage, path_gts, flair_prefix, mp2rage_prefix, gts_prefix, 
                            check=True,
                    transforms=None, num_workers=0, batch_size=1, cache_rate=0.5
                    ):
    """ Create a torch DataLoader based on the system arguments.
    """
    flair = sorted(glob(os.path.join(path_flair, f"*{flair_prefix}")),
                   key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    mp2rage = sorted(glob(os.path.join(path_mp2rage, f"*{mp2rage_prefix}")),
                   key=lambda i: int(re.sub('\D', '', i)))
    segs = sorted(glob(os.path.join(path_gts, f"*{gts_prefix}")),
                  key=lambda i: int(re.sub('\D', '', i)))
    segs_cl = sorted(glob(os.path.join(path_gts, "*cortical_lesions.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i)))
    
    if check:
        check_dataset([flair, mp2rage, segs, segs_cl], 
                      [flair_prefix, mp2rage_prefix, gts_prefix, 'cortical_lesions.nii.gz'])

    print(f"Initializing the dataset. Number of subjects {len(flair)}")

    files = [{"flair": fl, "mp2rage": mp2, "label": seg, "cl_mask": seg_cl} 
             for fl, mp2, seg, seg_cl in zip(flair, mp2rage, segs, segs_cl)]

    val_ds = CacheDataset(data=files, transform=transforms, 
                          cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(val_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)

def main(args):
    os.makedirs(args.path_save, exist_ok=True)
    
    subjects = []
    metrics_names = ['dice', 
                     'ppv', 'tpr', 'fpr',
                     'ppv_les_25', 'f1_les_25', 'fpr_les_25', 'tpr_les_25', 'fnr_les_25', 
                     'lfpr', 'ltpr']
    
    
    metrics_df = pd.DataFrame([], columns=metrics_names)
    metrics_df_cl = pd.DataFrame([], columns=metrics_names + ['present'])
    metrics_df_wm = pd.DataFrame([], columns=metrics_names + ['present'])
    
    for file in Path("/home/meri/ectrims/last_call/split2_data").glob("*_data.npy.npz"):
        npy_loader = np.load(file)
        seg_all=npy_loader['seg_all']
        gt_all=npy_loader['gt_all']
        gt_cl=npy_loader['gt_cl']
        gt_wml=npy_loader['gt_wml']
        all_outputs=npy_loader['all_outputs']
        
        filename_or_obj = os.path.basename(file)
        subjects.append(filename_or_obj)
            
        with Parallel(n_jobs=args.n_jobs) as parallel:
            # clean
            # seg_all = uncertainty_filtering(seg_all, all_outputs, 0.999999260172124, parallel)
            
            row = evaluate(seg_all, gt_all, None, parallel)
            metrics_df = metrics_df.append(row,ignore_index=True)
            metrics_df.to_csv(os.path.join(args.path_save, 'metrics.csv'))
            
            row = evaluate(seg_all, gt_cl, gt_wml, parallel)
            row['present'] = 1 if gt_cl.sum() >= 1 else 0
            metrics_df_cl = metrics_df_cl.append(row, ignore_index=True)
            metrics_df_cl.to_csv(os.path.join(args.path_save, 'metrics_cl.csv'))
            
            row = evaluate(seg_all, gt_wml, gt_cl, parallel)
            row['present'] = 1 if gt_wml.sum() >= 1 else 0
            metrics_df_wm = metrics_df_wm.append(row, ignore_index=True)
            metrics_df_wm.to_csv(os.path.join(args.path_save, 'metrics_wm.csv'))
        
    df_name = os.path.join(args.path_save, 'metrics_describe.csv')
    metrics_df.describe().to_csv(df_name)
    
    df_name = os.path.join(args.path_save, 'metrics.csv')
    metrics_df['subjects'] = subjects
    metrics_df.to_csv(df_name)
    
    for t, df in zip(['cl', 'wm'], [metrics_df_cl, metrics_df_wm]):
        df['subjects'] = subjects
        df_name = os.path.join(args.path_save, f'metrics_{t}.csv')
        df.to_csv(df_name)
        
        df = df[df['present'] == 1]
        
        df_name = os.path.join(args.path_save, f'metrics_{t}_describe.csv')
        df[[m for m in metrics_names]].describe().to_csv(df_name)
    
    logging.warning("False positive rate is only correctly computed for all lesions. This program does not alow lesion specific FPR computation")
    print(f"Saved results to folder {args.path_save}")

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)