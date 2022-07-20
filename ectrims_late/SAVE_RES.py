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
parser.add_argument('--path_save', type=str, default='/home/meri/ectrims/last_call/split2_data', 
                    help='Specify the path to save the segmentations')
parser.add_argument('--n_jobs', type=int, default=5,
                    help="number of cores used for computation of dsc for different retention fractions")


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
    
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    seed_val = 0
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    val_transforms_seed = Compose(
    [
        LoadNiftid(keys=["flair", "mp2rage", "label", 'cl_mask']),
        AddChanneld(keys=["flair", "mp2rage" ,"label", 'cl_mask']),
        Lambdad(keys=['cl_mask'], func=lambda x: (x > 0).astype(x.dtype)),
        Spacingd(keys=["flair", "mp2rage" ,"label", 'cl_mask'], 
                 pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear" ,"nearest", "nearest")),
        NormalizeIntensityd(keys=["flair", "mp2rage"], nonzero=True),
        ConcatItemsd(keys=["flair", "mp2rage"], name="image"),
        ToTensord(keys=["image", "label", 'cl_mask'])
    ]).set_random_state(seed=seed_val)

    val_loader = get_data_loader_with_cl(path_flair="/home/meri/data/wml_cl_detection/test",
                                 path_mp2rage="/home/meri/data/wml_cl_detection/test",
                                 path_gts="/home/meri/data/wml_cl_detection/test",
                                 flair_prefix="FLAIR.nii.gz",
                                 mp2rage_prefix="UNIT1.nii.gz",
                                 gts_prefix="all_lesions.nii.gz",
                                 transforms=val_transforms_seed,
                                 num_workers=10,
                                 batch_size=1, cache_rate=0.3)

    # %%
    print("Initializing the models")

    n_models = args.num_models

    models = []
    for i in range(n_models):
        models.append(UNet(
        dimensions=3,
        in_channels=2,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        norm='batch',
        num_res_units=0).to(device))

    act = torch.nn.Softmax(dim=1)

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

    # %%
    print("Staring evaluation")
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            # Get models predictions
            inputs, labels, mask_cl = (
                batch_data["image"].to(device),
                batch_data["label"],
                batch_data['cl_mask']
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
            # seg_all = np.squeeze(outputs_mean)      # [H, W, D]
            seg_all = remove_connected_components(segmentation=outputs_mean, l_min=args.lmin)
            seg_all[seg_all >= args.threshold] = 1.
            seg_all[seg_all < args.threshold] = 0.
            
            gt_all = np.squeeze(labels.cpu().numpy())
            gt_cl = np.squeeze(mask_cl.numpy())
            gt_wml = gt_all - gt_cl * gt_all
            
            filename_or_obj = batch_data['label_meta_dict']['filename_or_obj'][0]
            filename_or_obj = os.path.basename(filename_or_obj).split('.')[0] + '_data.npy'
            
            file = os.path.join(args.path_save, filename_or_obj)
            np.savez(file, seg_all=seg_all, gt_all=gt_all, gt_cl=gt_cl, gt_wml=gt_wml,
                     all_outputs=all_outputs)
            # %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)