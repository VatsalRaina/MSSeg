# -*- coding: utf-8 -*-

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


def get_connected_components(pred_map, uncs_map):
    # uncs_mask = uncs_map * brain_mask
    uncs_mask = (uncs_map > 0.01).astype("float") + (pred_map > 0.01).astype("float")
    uncs_mask[uncs_mask > 0] = 1.
    assert (np.unique(uncs_mask) == np.asarray([0, 1])).all(), f"{np.unique(uncs_mask)}"
    return ndimage.label(uncs_mask)[0]

def save_image(image, image_filepath, affine, original_affine, spatial_shape, mode):
    os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
    write_nifti(image,image_filepath,
                affine=affine,
                target_affine=original_affine,
                output_spatial_shape=spatial_shape, mode=mode)
    
def main(args):
    path_pred = os.path.join(args.path_save, "predictions")
    
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
    
    # %%
    print("Staring evaluation")
    npy_name = lambda x: re.sub("FLAIR", "data", x.split('.')[0]) + '.npz'
    
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            # Get models predictions
            inputs, gt = (
                batch_data["image"].to(device),
                batch_data["label"].cpu().numpy(),
                # batch_data["brain_mask"].cpu().numpy()
            )
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            all_outputs = []
            for model in models:
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, 
                                                   model, mode='gaussian')
                outputs = act(outputs).cpu().numpy()        # [1, 2, H, W, D]
                all_outputs.append(np.squeeze(outputs[0, 1]))
            all_outputs = np.asarray(all_outputs)        # [M, H, W, D]
            outputs_mean = np.mean(all_outputs, axis=0)     # [H, W, D]
            
            filename_or_obj = batch_data['image_meta_dict']['filename_or_obj']
            filename_or_obj = batch_data['image_meta_dict']['filename_or_obj']
            assert len(filename_or_obj) == 1
            filename_or_obj = os.path.basename(filename_or_obj[0])
            
            seg = outputs_mean.copy()
            seg[seg > args.threshold] = 1
            seg[seg < args.threshold] = 0
            unique = np.unique(seg)
            seg = np.squeeze(seg)      # [H, W, D]
            seg = remove_connected_components(segmentation=seg, l_min=9)
            if not (np.unique(seg) == np.array([0,1])).all():
                err_filename = os.path.join(args.path_save, "error", filename_or_obj)
                os.makedirs(os.path.join(args.path_save,  "error"))
                np.savez(err_filename, seg=seg, all_outputs=all_outputs)
                print(f"Found subject with weird predictions: unique labels {np.unique(seg)}\n"
                      f"Before removing connected components: {unique}\n"
                      f"Saved err seg to {err_filename}\n"
                      "Skipping subject")
                continue
            
            ens_seg = all_outputs.copy()
            ens_seg[ens_seg >= args.threshold] = 1.0
            ens_seg[ens_seg < args.threshold] = 0.0

            gt = np.squeeze(gt)     # [H, W, D]
            
            # Get all uncertainties
            uncs_value = ensemble_uncertainties_classification(
                np.concatenate((np.expand_dims(all_outputs, axis=-1),
                                np.expand_dims(1. - all_outputs, axis=-1)),
                               axis=-1))[args.unc_metric]   # [H, W, D]
            
            uncs_mask = get_connected_components(pred_map=outputs_mean, 
                                                 uncs_map=uncs_value)
            
            filename_or_obj = batch_data['image_meta_dict']['filename_or_obj']
            assert len(filename_or_obj) == 1
            filename_or_obj = os.path.basename(filename_or_obj[0])
            
            np.savez_compressed(os.path.join(args.path_save, npy_name(filename_or_obj)), 
                                all_outputs=all_outputs, gt=gt, seg=seg,
                                multi_uncs_mask=uncs_mask,
                                uncs_value=uncs_value)
            
            # metric_rf_row, f1_values = get_lesion_rc_with_fn(gts=gt,
            #                                          preds=seg,
            #                                          uncs=uncs_value,
            #                                          fracs_retained=fracs_ret,
            #                                          multi_uncs_mask=uncs_mask,
            #                                          IoU_threshold=args.IoU_threshold,
            #                                          n_jobs=args.n_jobs,
            #                                          all_outputs=ens_seg)
                        
            
            ###############################################################3
            
            # filename_or_obj = batch_data['image_meta_dict']['filename_or_obj'][0]
            # filename_or_obj = os.path.basename(filename_or_obj)
            # original_affine = batch_data['image_meta_dict']["original_affine"][0]
            # affine = batch_data['image_meta_dict']["affine"][0]
            # spatial_shape = batch_data['image_meta_dict']["spatial_shape"][0]
            
            # new_filename = re.sub(args.flair_prefix, 'uncs_mask.nii.gz',
            #                       filename_or_obj)
            # new_filepath = os.path.join(path_pred, "uncs_mask", new_filename)
            # save_image(uncs_mask, new_filepath, affine, original_affine, spatial_shape, 
            #            mode='bilinear')
            
            # new_filename = re.sub(args.flair_prefix, 'uncs_map.nii.gz',
            #                       filename_or_obj)
            # new_filepath = os.path.join(path_pred, "uncs_map", new_filename)
            # save_image(uncs_value, new_filepath, affine, original_affine, spatial_shape,
            #            mode='nearest')
            
            # new_filename = re.sub(args.flair_prefix, 'prob_pred.nii.gz',
            #                       filename_or_obj)
            # new_filepath = os.path.join(path_pred, "prob_pred", new_filename)
            # save_image(outputs_mean, new_filepath, affine, original_affine, spatial_shape,
            #            mode='bilinear')
            
            # new_filename = re.sub(args.flair_prefix, 'pred.nii.gz',
            #                       filename_or_obj)
            # new_filepath = os.path.join(path_pred, "pred", new_filename)
            # save_image(seg, new_filepath, affine, original_affine, spatial_shape,
            #            mode='nearest')
            
            ####################################################################

           
# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
