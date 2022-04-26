#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
         Adapted by Nataliia Molchanova
         
runfile('/home/meri/Code/MSSeg/nataliia/uncertainty_based_filtering.py', wdir='/home/meri/Code/MSSeg/', args='--threshold 0.35 --num_models 5 --path_data /home/meri/data/canonical/dev_in --path_gts /home/meri/data/canonical/dev_in --path_model /home/meri/uncertainty_challenge/models_cv/ex_ljubljana/ --flair_prefix FLAIR.nii.gz --gts_prefix gt.nii --check_dataset --output_file /home/meri/uncertainty_challenge/results/uncertainty_threshold/ --n_jobs 10 --num_workers 8')
"""

import argparse
import os
import torch
import re
from glob import glob
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld, Compose, CropForegroundd, LoadNiftid, Orientationd, RandCropByPosNegLabeld,
    ScaleIntensityRanged, Spacingd, ToTensord, ConcatItemsd, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd, AsDiscrete, Activations)
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from one_modality.Uncertainty import ensemble_uncertainties_classification
import pandas as pd

from one_modality.utils.data_load import *
from one_modality.utils.setup import get_default_device
from one_modality.utils.metrics import get_dsc_norm_auc

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
# parser.add_argument('--path_save', type=str, default='', help='Specify the path to save the segmentations')
parser.add_argument('--flair_prefix', type=str, default="FLAIR.nii.gz", help='name ending FLAIR')
parser.add_argument('--gts_prefix', type=str, default="gt.nii", help='name ending segmentation mask')
parser.add_argument('--output_file', type=str, default='', required=True,
                    help="Specifily the file to store the results")
parser.add_argument('--check_dataset', action='store_true',
                    help='if sent, checks that FLAIR and semg masks names correspond to each other')
parser.add_argument('--n_jobs', type=int, default=1,
                    help="number of cores used for computation of dsc for different retention fractions")
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of jobs used to load the data, DataLoader parameter')


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
            pthfilepath = os.path.join(args.path_model, "seed" + str(i+1), "Best_model_finetuning.pth")
            model.load_state_dict(torch.load(pthfilepath))
            model.eval()
    elif n_models == 1:
        pthfilepath = os.path.join(args.path_model, "seed" + str(args.i_seed), "Best_model_finetuning.pth")
        models[0].load_state_dict(torch.load(pthfilepath))
        models[0].eval()
    else:
        raise ValueError(f"invalid number of num_models {args.num_models}")

    # %%
    print("Starting computations")
    uncs_value_global = []
    max_uncs = []
    min_uncs = []
    num_patients = 0
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
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
                outputs = act(outputs).cpu().numpy()
                all_outputs.append(np.squeeze(outputs[0, 1]))
            all_outputs = np.asarray(all_outputs)
            outputs_mean = np.mean(all_outputs, axis=0)

            # Get all uncertainties
            uncs_value = ensemble_uncertainties_classification(
                np.concatenate((np.expand_dims(all_outputs, axis=-1), np.expand_dims(1. - all_outputs, axis=-1)),
                               axis=-1))[args.unc_metric]
            uncs_value_global.append(uncs_value.flatten()[gt.cpu().numpy().flatten() > 0])
            max_uncs.append(np.max(uncs_value))
            min_uncs.append(np.min(uncs_value))
    
    uncs_value_global = np.concatenate(uncs_value_global, axis=0)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(file=args.output_file, arr=uncs_value_global)
    
    plt.hist(uncs_value_global, bins=100)
    plt.title(f"Distribution of {args.unc_metric} in in_dev voxels")
    plt.savefig(os.path.join(os.path.dirname(args.output_file), 'histogram.jpg'))
    plt.clf()
    
    plt.hist(uncs_value_global, bins=100)
    plt.title(f"Log distribution of {args.unc_metric} in in_dev voxels")
    plt.yscale("log")
    plt.savefig(os.path.join(os.path.dirname(args.output_file), 'histogram_log.jpg'))
    plt.clf()
    
    plt.hist(max_uncs, bins=len(max_uncs))
    plt.title(f"Max value {args.unc_metric} in in_dev scans")
    plt.savefig(os.path.join(os.path.dirname(args.output_file), 'histogram_max.jpg'))
    plt.clf()
    
    plt.hist(min_uncs, bins=len(min_uncs))
    plt.title(f"Min {args.unc_metric} in in_dev scans")
    plt.savefig(os.path.join(os.path.dirname(args.output_file), 'histogram_min.jpg'))
    plt.clf()
    
    percent = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
    quantiles = np.quantile(uncs_value_global, q=percent)
    pd.DataFrame(np.expand_dims(quantiles, axis=0), columns=percent).to_csv(os.path.join(os.path.dirname(args.output_file), 'quantiles.csv'))
    
    plt.hist(uncs_value_global, bins=100)
    plt.title(f"Log distribution of {args.unc_metric} in in_dev voxels\nwith quantiles")
    plt.yscale("log")
    for p, q in enumerate(quantiles):
        plt.plot([q, q], [0, 1e7], '--',label=f'{percent[p]} quantile')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(args.output_file), 'histogram_log_quantiles.jpg'))
    plt.clf()
    
# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
