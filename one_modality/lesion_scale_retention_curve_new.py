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
from utils.metrics import get_lesion_rc
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
parser.add_argument('--IoU_threshold', default=0.5, type=float, help="IoU threshold for lesion F1 computation")


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
    thresh_str = "%.3d" % (args.IoU_threshold * 100)
    
    # %%
    print("Staring evaluation")
    num_patients = 0
    fracs_ret = np.log(np.arange(200 + 1)[1:])
    fracs_ret /= np.amax(fracs_ret)
    uncs_metrics = ['sum', 'mean', 'logsum', 'median', 'volume']
    uncs_metrics = uncs_metrics + [um + '_ext' for um in uncs_metrics]
    metric_rf = dict(zip(
        uncs_metrics, [pd.DataFrame([], columns=fracs_ret) for _ in uncs_metrics]
    ))
    f1_dict = dict()
    
    
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            # Get models predictions
            num_patients += 1
            inputs, gt = (
                batch_data["image"].to(device),
                batch_data["label"].cpu().numpy(),
                # batch_data["brain_mask"].cpu().numpy()
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
            
            seg = outputs_mean.copy()
            seg[seg > args.threshold] = 1
            seg[seg < args.threshold] = 0
            seg = np.squeeze(seg)      # [H, W, D]
            seg = remove_connected_components(segmentation=seg, l_min=9)
            assert (np.unique(seg) == np.array([0,1])).all(), f"{np.unique(seg)}"

            gt = np.squeeze(gt)     # [H, W, D]
            
            meta_data = batch_data['image_meta_dict']
            for i, data in enumerate(outputs_o):  
                out_meta = {k: meta_data[k][i] for k in meta_data} if meta_data else None

            filename_or_obj = out_meta.get("filename_or_obj", None) if out_meta else None

            # Get all uncertainties
            uncs_value = ensemble_uncertainties_classification(
                np.concatenate((np.expand_dims(all_outputs, axis=-1),
                                np.expand_dims(1. - all_outputs, axis=-1)),
                               axis=-1))[args.unc_metric]   # [H, W, D]
            
            uncs_mask = get_connected_components(pred_map=outputs_mean, 
                                                 uncs_map=uncs_value)
            
            ###############################################################3
            
            filename_or_obj = batch_data['image_meta_dict']['filename_or_obj'][0]
            filename_or_obj = os.path.basename(filename_or_obj)
            original_affine = batch_data['image_meta_dict']["original_affine"][0]
            affine = batch_data['image_meta_dict']["affine"][0]
            spatial_shape = batch_data['image_meta_dict']["spatial_shape"][0]
            
            new_filename = re.sub(args.flair_prefix, 'uncs_mask.nii.gz',
                                  filename_or_obj)
            new_filepath = os.path.join(path_pred, "uncs_mask", new_filename)
            save_image(uncs_mask, new_filepath, affine, original_affine, spatial_shape, 
                       mode='bilinear')
            
            new_filename = re.sub(args.flair_prefix, 'uncs_map.nii.gz',
                                  filename_or_obj)
            new_filepath = os.path.join(path_pred, "uncs_map", new_filename)
            save_image(uncs_value, new_filepath, affine, original_affine, spatial_shape,
                       mode='nearest')
            
            new_filename = re.sub(args.flair_prefix, 'prob_pred.nii.gz',
                                  filename_or_obj)
            new_filepath = os.path.join(path_pred, "prob_pred", new_filename)
            save_image(outputs_mean, new_filepath, affine, original_affine, spatial_shape,
                       mode='bilinear')
            
            new_filename = re.sub(args.flair_prefix, 'pred.nii.gz',
                                  filename_or_obj)
            new_filepath = os.path.join(path_pred, "pred", new_filename)
            save_image(seg, new_filepath, affine, original_affine, spatial_shape,
                       mode='nearest')
            
            ####################################################################

            metric_rf_row, f1_values = get_lesion_rc(gts=gt,
                                                     preds=seg,
                                                     uncs=uncs_value,
                                                     fracs_retained=fracs_ret,
                                                     multi_uncs_mask=uncs_mask,
                                                     IoU_threshold=args.IoU_threshold,
                                                     n_jobs=args.n_jobs)
            for ut in uncs_metrics:
                row_df = pd.DataFrame(np.expand_dims(metric_rf_row[ut], axis=0), 
                                      columns=fracs_ret, index=[0])
                metric_rf[ut] = metric_rf[ut].append(row_df, ignore_index=True)
                
            f1_dict[os.path.basename(filename_or_obj)] = f1_values
            
            pkl_filepath = os.path.join(args.path_save, f"subject_f1{thresh_str}_{args.unc_metric}.pkl")
            with open(pkl_filepath, 'wb') as f:
                pickle.dump(f1_dict, f)
                
            pkl_filepath = os.path.join(args.path_save, f"RC_f1{thresh_str}_{args.unc_metric}.pkl")
            with open(pkl_filepath, 'wb') as f:
                pickle.dump(metric_rf, f)
            
            if num_patients % 10 == 0:
                print(f"Processed {num_patients} scans")
    
    with open(os.path.join(args.path_save, f"f1{thresh_str}-AUC_{args.unc_metric}.csv"), 'w') as f:
        for ut in uncs_metrics:
            mean_aac = metrics.auc(fracs_ret, np.asarray(metric_rf[ut].mean()))
            f.write(f"{ut} AAC:\t{mean_aac}")
    
            plot_iqr_median_rc(metric_rf[ut], fracs_ret,
                               os.path.join(args.path_save, 
                                            f"IQR_RC_f1{thresh_str}_{args.unc_metric}_{ut}_df.png"))
            plot_mean_rc(metric_rf[ut], fracs_ret,
                               os.path.join(args.path_save, 
                                            f"mean_RC_f1{thresh_str}_{args.unc_metric}_{ut}_df.png"))
            for i, row in metric_rf[ut].iterrows():
                plt.plot(fracs_ret, row)
            plt.savefig(os.path.join(args.path_save, f"subject_RC_f1{thresh_str}_{args.unc_metric}_{ut}_df.png"))
            plt.close('all')
            
    save_path = os.path.join(args.path_save, 
                             f"mean_RC_f1{thresh_str}_{args.unc_metric}_df.png")
    for ut in uncs_metrics:
        mean = metric_rf[ut].mean()
        plt.plot(fracs_ret, mean, label=ut)

    plt.xlim([0, 1.01])
    plt.legend()
    plt.xlabel("Retention fraction")
    plt.ylabel("F1 lesion-scale")
    plt.savefig(save_path)
    plt.clf()
    
    print(f"Saved f1 scores and retention curve to folder {args.path_save}")

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
