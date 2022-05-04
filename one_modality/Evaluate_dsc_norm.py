"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
         Adapted by Nataliia Molchanova
         
         Compute average dice score for the model.
         
         python one_modality/Process_dsc_norm_tmp.py \
		--threshold 0.35 \
		--num_models 5 \
		--path_data "/home/meri/data/msseg_advanced/FLAIR_preproc" \
		--path_gts "/home/meri/data/msseg_advanced/lesion_mask" \
		--path_model "/home/meri/uncertainty_challenge/models_cv/ex_ljubljana/" \
		--flair_prefix "FLAIR.nii.gz" \
		--gts_prefix "labeled_locations_lesions.nii.gz" \
		--check_dataset \
		--output_file "/home/meri/uncertainty_challenge/results_advanced/Process_dsc_norm_tmp/dsc_norm_Lausanne.npz" \
        --n_jobs 10 --num_workers 10
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
from Uncertainty import ensemble_uncertainties_classification

from utils.data_load import *
from utils.setup import get_default_device
from utils.metrics import dice_norm_metric

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for lesion detection')
parser.add_argument('--patient_num', type=int, default=0, help='Specify 1 patient to permit parallel processing')
parser.add_argument('--num_models', type=int, default=5, help='Number of models in ensemble')
parser.add_argument('--i_seed', type=int, default=-1,
                    help='if num_models==1, it will use the i_seed model for predictions')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_model', type=str, default='', help='Specify the dir to al the trained models')
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
    print("Computing DSC_norm")
    dsc_norm_value = 0
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

            outputs_mean[outputs_mean > args.threshold] = 1
            outputs_mean[outputs_mean < args.threshold] = 0
            seg = np.squeeze(outputs_mean)
            seg = remove_connected_components(segmentation=seg, l_min=9)

            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)

            dsc_norm = dice_norm_metric(gt.flatten(), seg.flatten())[0]
            # print("auc", auc_dsc_norm)
            dsc_norm_value += dsc_norm
            
            if num_patients % 10 == 0:
                print(f"Processed {num_patients}")

    # compute the average
    dsc_norm_value /= num_patients
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, 'w') as f:
        f.write(f"mean_dsc_norm\t{dsc_norm_value}")

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
