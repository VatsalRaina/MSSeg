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
from joblib import Parallel, delayed
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
from collections import Counter

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--les_unc_metric', type=str, 
                    default="reverse_mutual_information",
                    help='name of a single uncertainty metric')
parser.add_argument('--path_les_csv', type=str, default='', 
                    help='Specify the path to the test data files directory')
parser.add_argument('--path_fn_csv', type=str, default='', 
                    help='Specify the path to the test data files directory')
parser.add_argument('--path_save', type=str, default='', 
                    help='Specify the path to save the segmentations')
parser.add_argument('--n_jobs', type=int, default=1,
                    help="number of cores used for computation of dsc for different retention fractions")

def compute_f1(lesion_types, fn_count):
        counter = Counter(lesion_types)
        if counter['tp'] + 0.5 * (counter['fp'] + fn_count) == 0.0:
            return 0
        return counter['tp'] / (counter['tp'] + 0.5 * (counter['fp'] + fn_count))
     
def get_unique_hash(les_df, fn_df, hash_col='hash'):
    les_hash = set(les_df[hash_col])
    fn_hash = set(fn_df[hash_col])
    
    assert len(les_hash.intersection(fn_hash)) == len(les_hash) == len(fn_hash)
    
    return les_hash
        
def main(args):
    les_uncs_df = pd.read_csv(args.path_les_csv, index_col=0)
    n_fn_df = pd.read_csv(args.path_fn_csv, index_col=0)
    hashes = get_unique_hash(les_uncs_df, n_fn_df)
    n_fn_df = n_fn_df.set_index('hash')
    lum = args.les_unc_metric
        
    path_save = args.path_save
    os.makedirs(args.path_save, exist_ok=True)
    
    print("Staring evaluation")
    
    # Sort uncertainties
    les_uncs_df = les_uncs_df.sort_values(by=lum, ascending=False)
    les_uncs_mod_df = les_uncs_df.copy()
    
    res_df = pd.DataFrame([], columns=['mean F1', args.les_unc_metric, 'hash'])
    # Excluding the most uncertain lesions one by one
    for i_l, row in les_uncs_df.iterrows():
        if row['type'] == 'fp':
            les_uncs_mod_df.loc[i_l, 'type'] = 'tn'
        else: # only fp anf tp lesions
            les_uncs_mod_df.loc[i_l, 'type'] = 'fn'
            n_fn_df.loc[row['hash'],'fn_count'] += 1
            
        # compute average of lesion-wise uncertainty from lesions dataframe
        f1_scores = Parallel(n_jobs=args.n_jobs)(delayed(compute_f1)(
            lesion_types=list(les_uncs_mod_df[les_uncs_mod_df['hash'] == h]['type']),
            fn_count=n_fn_df.loc[h, 'fn_count']
        ) for h in hashes)
        
        res_df = res_df.append({'mean F1': np.mean(f1_scores), 
                                args.les_unc_metric: row[args.les_unc_metric],
                                'hash': row['hash']},
                               ignore_index=True)
        res_df.to_csv(os.path.join(path_save, f"{args.les_unc_metric}_threshold_tuning.csv"))

    print("The best fn corresponds to the setting:")
    print(res_df.sort_values(by='mean F1', ascending=False).head(1))
    print(f"Saved resutls to folder {args.path_save}")

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
