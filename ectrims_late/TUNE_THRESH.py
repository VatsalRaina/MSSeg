# -*- coding: utf-8 -*-
import argparse
import os
from functools import partial
import random
import numpy as np
from pathlib import Path
from glob import glob
import torch
from joblib import Parallel, delayed
import re
from monai.data import write_nifti
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    AddChanneld, Compose, CropForegroundd, LoadNiftid, Orientationd, RandCropByPosNegLabeld,
    ScaleIntensityRanged, Spacingd, ToTensord, ConcatItemsd, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd, AsDiscrete, Activations,
    Lambdad)
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
from utils.retention_curve import detection_uncertainty
from utils.visualise import plot_iqr_median_rc, plot_mean_rc
from utils.transforms import get_TP_FP_lesions_mask_parallel, get_FN_lesions_count_parallel
from collections import Counter


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--path_save', type=str, default='/home/meri/ectrims/last_call/split1_filtering', 
                    help='Specify the path to save the segmentations')
parser.add_argument('--n_jobs', type=int, default=5,
                    help="number of cores used for computation of dsc for different retention fractions")
parser.add_argument('--threshold', type=float, default=0.4, help='Threshold for lesion detection')
parser.add_argument('--lmin', type=int, default=6,
                    help="minimal size of predicted lesions")


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


def get_data_loader_with_cl(path_flair, path_mp2rage, path_gts, flair_prefix, mp2rage_prefix, gts_prefix, check=True,
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
    path_save = args.path_save
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

    val_loader = get_data_loader_with_cl(path_flair="/home/meri/data/wml_cl_detection/val",
                                 path_mp2rage="/home/meri/data/wml_cl_detection/val",
                                 path_gts="/home/meri/data/wml_cl_detection/val",
                                 flair_prefix="FLAIR.nii.gz",
                                 mp2rage_prefix="UNIT1.nii.gz",
                                 gts_prefix="all_lesions.nii.gz",
                                 transforms=val_transforms_seed,
                                 num_workers=5,
                                 batch_size=1, cache_rate=0.1)

    # %%
    print("Initializing the models")

    n_models = 3

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
    
    path_model = "/home/meri/ectrims/cluster_training/wml_cl_detection_train"

    if n_models > 1:
        for i, model in enumerate(models):
            pthfilepath = os.path.join(path_model, "seed" + str(i + 1), "Best_model_finetuning.pth")
            model.load_state_dict(torch.load(pthfilepath))
            model.eval()
    elif n_models == 1:
        pthfilepath = os.path.join(path_model, "seed" + str(args.i_seed), "Best_model_finetuning.pth")
        models[0].load_state_dict(torch.load(pthfilepath))
        models[0].eval()
    else:
        raise ValueError(f"invalid number of num_models {n_models}")

    # %%
    print("Staring evaluation")    
    les_uncs_df = pd.DataFrame([], columns=['hash', 'type', 'number', 'det uncs'])
    n_fn_df = pd.DataFrame([], columns=['hash', 'fn_count'])
    
    # save uncertainty of all tp and fp lesions, and save the number of fn lesions
    
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            # Get models predictions
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"]
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

            filename_or_obj = batch_data['label_meta_dict']['filename_or_obj'][0]
            filename_or_obj = os.path.basename(filename_or_obj)
            
            with Parallel(n_jobs=args.n_jobs) as parallel:
                n_fn = get_FN_lesions_count_parallel(ground_truth=gt_all, 
                                                   predictions=seg_all, 
                                                   IoU_threshold=0.25, 
                                                   mask_type='binary', 
                                                   parallel_backend=parallel)
        
                tp_lesions, fp_lesions = get_TP_FP_lesions_mask_parallel(ground_truth=gt_all,
                                                          predictions=seg_all,
                                                          IoU_threshold=0.25,
                                                          mask_type='binary',
                                                          parallel_backend=parallel)
                tp_uncs = detection_uncertainty(tp_lesions, all_outputs, parallel)
                fp_uncs = detection_uncertainty(fp_lesions, all_outputs, parallel)
            
            n_fn_row = pd.DataFrame({'fn_count': n_fn, 'hash': filename_or_obj},
                                    index=[0])
            n_fn_df = pd.concat([n_fn_df, n_fn_row], axis=0, ignore_index=True)
            
            for i_les, les_uncs in enumerate(tp_uncs):
                les_row = pd.DataFrame({'hash':filename_or_obj, 
                                        'type': 'tp', 
                                        'number': i_les, 
                                        'det uncs': les_uncs}, index=[0])
                les_uncs_df = pd.concat([les_uncs_df, les_row], axis=0, ignore_index=True)
                
            for i_les, les_uncs in enumerate(fp_uncs):
                les_row = pd.DataFrame({'hash':filename_or_obj, 
                                        'type': 'fp', 
                                        'number': i_les, 
                                        'det uncs': les_uncs}, index=[0])
                les_uncs_df = pd.concat([les_uncs_df, les_row], axis=0, ignore_index=True)
                
            n_fn_df.to_csv(os.path.join(path_save, "fn_count.csv"))
            les_uncs_df.to_csv(os.path.join(path_save, "det_uncs.csv"))
                
    hashes = get_unique_hash(les_uncs_df, n_fn_df)
    n_fn_df = n_fn_df.set_index('hash')
    lum = 'det uncs'
    
    print("Staring evaluation")
    
    # Sort uncertainties
    les_uncs_df = les_uncs_df.sort_values(by=lum, ascending=False)
    les_uncs_mod_df = les_uncs_df.copy()
    
    res_df = pd.DataFrame([], columns=['mean F1', lum, 'hash'])
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
                                lum: row[lum],
                                'hash': row['hash']},
                               ignore_index=True)
        res_df.to_csv(os.path.join(path_save, f"{lum}_threshold_tuning.csv"))

    print("The best fn corresponds to the setting:")
    print(res_df.sort_values(by='mean F1', ascending=False).head(1))
    print(f"Saved resutls to folder {args.path_save}")


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
