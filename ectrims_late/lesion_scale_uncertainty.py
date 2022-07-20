"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
         Adapted by Nataliia Molchanova
         
         
         
runfile('/home/meri/Code/MSSeg/one_modality/lesion_scale_uncertainty.py', wdir='/home/meri/Code/MSSeg/one_modality', args='--threshold 0.35 --num_models 5 --path_data /home/meri/data/canonical/dev_in --path_gts /home/meri/data/canonical/dev_in --path_model /home/meri/uncertainty_challenge/models_cv/ex_ljubljana/ --path_save /home/meri/uncertainty_challenge/results/f1_lesion_score/ --flair_prefix FLAIR.nii.gz --gts_prefix gt.nii --check_dataset --n_jobs 10 --num_workers 10')
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
import json

sns.set_theme()
from Uncertainty import ensemble_uncertainties_classification, lesions_uncertainty_sum
import pandas as pd
from utils.data_load import *
from utils.setup import get_default_device
from utils.transforms import get_FN_lesions_mask_parallel, get_TP_FP_lesions_mask_parallel
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


def compute_lesion_uncs(lesion, uncs_map):
    return np.sum(lesion * uncs_map)/np.sum(lesion)
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

    # %%
    print("Staring evaluation")
    num_patients = 0
    IoU_threshold = 0.5
    
    subject_uncs = dict()
    
    fn_uncs_all, fp_uncs_all, tp_uncs_all = [], [], []
    
    fig, ax = plt.subplots(len(val_loader), 1, figsize=(4, 4*len(val_loader)))
    
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
                outputs_o = (act(outputs))
                outputs = act(outputs).cpu().numpy()        # [1, 2, H, W, D]
                all_outputs.append(np.squeeze(outputs[0, 1]))
            all_outputs = np.asarray(all_outputs)        # [M, H, W, D]
            outputs_mean = np.mean(all_outputs, axis=0)     # [H, W, D]

            outputs_mean[outputs_mean > args.threshold] = 1
            outputs_mean[outputs_mean < args.threshold] = 0
            seg = np.squeeze(outputs_mean)      # [H, W, D]
            seg = remove_connected_components(segmentation=seg, l_min=3)
            seg[seg >= args.threshold] = 1.
            seg[seg < args.threshold] = 0.

            val_labels = gt.cpu().numpy()   # [1, 1, H, W, D]
            gt = np.squeeze(val_labels)     # [H, W, D]
            #gt = remove_connected_components(segmentation=gt, l_min=9)

            # Get all uncertainties
            uncs_value = ensemble_uncertainties_classification(
                np.concatenate((np.expand_dims(all_outputs, axis=-1),
                                np.expand_dims(1. - all_outputs, axis=-1)),
                               axis=-1))[args.unc_metric]   # [H, W, D]
            
            meta_data = batch_data['image_meta_dict']
            for i, data in enumerate(outputs_o):  
                out_meta = {k: meta_data[k][i] for k in meta_data} if meta_data else None

            filename_or_obj = out_meta.get("filename_or_obj", None) if out_meta else None
            
            process = partial(compute_lesion_uncs, uncs_map=uncs_value)
            
            with Parallel(n_jobs=args.n_jobs) as parallel:
                tp_lesions, fp_lesions = get_TP_FP_lesions_mask_parallel(ground_truth=gt, 
                                                                         predictions=seg, 
                                                                         IoU_threshold=IoU_threshold, 
                                                                         mask_type='one_hot', 
                                                                         parallel_backend=parallel)
                fn_lesions = get_FN_lesions_mask_parallel(ground_truth=gt, 
                                                          predictions=seg, 
                                                          IoU_threshold=IoU_threshold, 
                                                          mask_type='one_hot', 
                                                          parallel_backend=parallel)
                
                fn_uncs = parallel(delayed(process)(les) for les in fn_lesions)
                tp_uncs = parallel(delayed(process)(les) for les in tp_lesions)
                fp_uncs = parallel(delayed(process)(les) for les in fp_lesions)
                
                subject_uncs[os.path.basename(filename_or_obj)] = {
                    "fn": fn_uncs, "tp": tp_uncs, "fp": fp_uncs
                    }
            
            # ax[count].hist(fn_uncs, bins=len(fn_uncs), color='red', label='FN', alpha=0.5)
            # ax[count].hist(tp_uncs, bins=len(tp_uncs),color='green', label='TP', alpha=0.5)
            # ax[count].hist(fp_uncs, bins=len(fp_uncs),color='blue', label='FP', alpha=0.5)
            # ax[count].set_title(os.path.basename(filename_or_obj))
            # ax[count].legend()
            
            fn_uncs_all.extend(fn_uncs)
            tp_uncs_all.extend(tp_uncs)
            fp_uncs_all.extend(fp_uncs)
            
            if num_patients % 10 == 0:
                print(f"Processed {num_patients} scans")

    os.makedirs(args.path_save, exist_ok=True)
    thresh_str = "%.3d" % (IoU_threshold * 100)
    plot_filename = os.path.join(args.path_save, "uncs_plot_subjects.jpg")
    fig.savefig(plot_filename)
    plt.clf()
    
    plot_filename = os.path.join(args.path_save, "uncs_plot_all.jpg")
    plt.hist(fn_uncs_all, bins=min(len(fn_uncs_all), 100), label='FN', color='red', alpha=0.5)
    plt.hist(tp_uncs_all, bins=min(len(tp_uncs_all), 100), label='TP', color='green', alpha=0.5)
    plt.hist(fp_uncs_all, bins=min(len(fp_uncs_all), 100), label='FP', color='blue', alpha=0.5)
    plt.legend()
    plt.savefig(plot_filename)
    plt.clf()
    
    np.save(os.path.join(args.path_save, "fn.npy"), np.asarray(fn_uncs_all))
    np.save(os.path.join(args.path_save, "tp.npy"), np.asarray(tp_uncs_all))
    np.save(os.path.join(args.path_save, "fp.npy"), np.asarray(fp_uncs_all))
    
    json_filepath = os.path.join(args.path_save, "subject_uncs.json")
    with open(json_filepath, 'w') as f:
        json.dump(subject_uncs, f)
    
    print(f"Saved results to folder {args.path_save}")

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
