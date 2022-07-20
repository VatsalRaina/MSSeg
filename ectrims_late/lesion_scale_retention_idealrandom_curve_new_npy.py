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
import json

sns.set_theme()
from Uncertainty import ensemble_uncertainties_classification
import pandas as pd
from scipy import ndimage
from utils.data_load import remove_connected_components, get_data_loader
from utils.setup import get_default_device
from utils.retention_curve import get_lesion_rc_with_fn
from utils.visualise import plot_iqr_median_rc, plot_mean_rc
from collections import Counter
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--path_data', type=str, default='', 
                    help='Specify the path to the test data files directory')
parser.add_argument('--path_save', type=str, default='', 
                    help='Specify the path to save the segmentations')

def compute_f1(lesion_types):
        counter = Counter(lesion_types)
        if counter['tp'] + 0.5 * (counter['fp'] + counter['fn']) == 0.0:
            return 0
        return counter['tp'] / (counter['tp'] + 0.5 * (counter['fp'] + counter['fn']))
    
def compute_rc(ordering, lesion_types, fracs_retained):
    lesion_types_copy = lesion_types.copy()
    lesion_types_copy = lesion_types_copy[ordering][::-1]

    f1_values = [compute_f1(lesion_types_copy)]

    # reject the most uncertain lesion
    for i_l, lesion_type in enumerate(lesion_types_copy):
        if lesion_type == 'fp':
            lesion_types_copy[i_l] = 'tn'
        elif lesion_type == 'fn':
            lesion_types_copy[i_l] = 'tp'
        f1_values.append(compute_f1(lesion_types_copy))

    # interpolate the curve and make predictions in the retention fraction nodes
    n_lesions = lesion_types_copy.shape[0]
    spline_interpolator = interp1d(x=[_ / n_lesions for _ in range(n_lesions + 1)], 
                                   y=f1_values[::-1],
                                   kind='slinear', fill_value="extrapolate")
    return spline_interpolator(fracs_retained), f1_values[::-1]
    
def main(args):
    npy_files = Path(args.path_data).glob("*lesion_data.npz")
        
    os.makedirs(args.path_save, exist_ok=True)
    path_save = args.path_save
    
    # %%
    print("Staring evaluation")
    fracs_ret = np.log(np.arange(200 + 1)[1:])
    fracs_ret /= np.amax(fracs_ret)
    
    random_df = pd.DataFrame([], columns=fracs_ret)
    ideal_df = random_df.copy()
    
    f1_random = dict()
    f1_ideal = dict()
    
    subjects = []
    
    np.random.seed(1)
    
    for npy_file in npy_files:
        # loading variables
        npy_loader = np.load(npy_file)
        
        filename_or_obj = re.sub("lesion_data.npz", "", os.path.basename(npy_file))
        subjects.append(filename_or_obj)
        
        n_tp = ndimage.label(npy_loader['tp_lesions'])[1]
        n_fp = ndimage.label(npy_loader['fp_lesions'])[1]
        n_fn = npy_loader['n_fn']
        
        lesion_types = [np.full(shape=(n,), fill_value=t) 
                         for n,t in zip([n_tp, n_fp, n_fn], ['tp', 'fp', 'fn'])]
        lesion_types = np.concatenate(lesion_types, axis=0)
        
        np.random.shuffle(lesion_types)
        
        random_uncs = np.random.rand(len(lesion_types))
        ideal_uncs = np.zeros_like(lesion_types, dtype=float)
        ideal_uncs[np.where(lesion_types == 'fp')] += 1
        ideal_uncs[np.where(lesion_types == 'fn')] += 2
        
        rand_rc, random_f1 = compute_rc(ordering=random_uncs.argsort(), 
                                        lesion_types=lesion_types, 
                                        fracs_retained=fracs_ret)
        ideal_rc, ideal_f1 = compute_rc(ordering=ideal_uncs.argsort(), 
                                        lesion_types=lesion_types, 
                                        fracs_retained=fracs_ret)
        
        random_df = random_df.append(pd.DataFrame(np.expand_dims(rand_rc, axis=0), columns=fracs_ret), 
                                     ignore_index=True)
        ideal_df = ideal_df.append(pd.DataFrame(np.expand_dims(ideal_rc, axis=0), columns=fracs_ret), 
                                   ignore_index=True)
        
        f1_random[filename_or_obj] = random_f1
        f1_ideal[filename_or_obj] = ideal_f1
        
        random_df.to_csv(os.path.join(path_save, "random_rc.csv"))
        ideal_df.to_csv(os.path.join(path_save, "ideal_rc.csv"))
        
        with open(os.path.join(path_save, "f1_random.json"), 'w') as fj:
            json.dump(f1_random, fj)
        with open(os.path.join(path_save, "f1_ideal.json"), 'w') as fj:
            json.dump(f1_ideal, fj)
            
    random_df['hash'] = subjects
    ideal_df['hash'] = subjects
    random_df.to_csv(os.path.join(path_save, "random_rc.csv"))
    ideal_df.to_csv(os.path.join(path_save, "ideal_rc.csv"))    
    print(f"Saved f1 scores and retention curve to folder {args.path_save}")

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
