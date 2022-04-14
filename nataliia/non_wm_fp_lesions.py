#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:52:33 2022

@author: Nataliia Molchanova

Find intersections between the predictions of ex_ljubljana and non-WM lesions
"""
import os
import nibabel as nib
import numpy as np
from monai.transforms import AddChannel, Spacing


root_dir = "/home/meri/data/msseg_advanced/"
save_dir = "/home/meri/uncertainty_challenge/results_advanced/non_wm_lesion"

model_pred_dirname = "FLAIR_pred_ex_ljubljana"
pred_prefix = "labeled_locations_lesions_pred.nii.gz"

masks_list = ['lesion_mask_deep_GM', 'lesion_mask_brainstem', 'lesion_mask_CL']
gt_prefix = "labeled_locations_lesions.nii.gz"

if __name__ == '__main__':
    masks_intersec = dict(zip(masks_list, [0 for _ in masks_list]))
    os.makedirs(save_dir, exist_ok=True)
    
    for sub_num in range(21, 66):
        sub = 'sub-%.3d' % sub_num
        for ses_num in [1, 2]:
            ses = 'ses-%.2d' % ses_num
            pred_mask_filepath = os.path.join(root_dir, model_pred_dirname, f"{sub}_{ses}_{pred_prefix}")
            if os.path.exists(pred_mask_filepath):
                pred_mask = nib.load(
                        pred_mask_filepath
                        ).get_fdata()
                for mask_dirname in masks_list:
                    gt_type_filepath = os.path.join(root_dir, mask_dirname, f"{sub}_{ses}_{pred_prefix}")
                    
                    if os.path.exists(gt_type_filepath):
                        gt_type_mask = nib.load(
                            gt_type_filepath
                            )
                        gt_affine = gt_type_mask.affine
                        gt_type_mask = gt_type_mask.get_fdata()
                        gt_type_mask_tr = AddChannel()(gt_type_mask)
                        gt_type_mask_tr = Spacing(
                            pixdim=(1.0, 1.0, 1.0), mode="nearest")(
                                gt_type_mask_tr, affine=gt_affine
                                )[0][0]
                        
                        if gt_type_mask_tr.shape != pred_mask.shape:
                            print(gt_type_filepath)
                        
                        if np.sum(pred_mask * gt_type_mask_tr) > 1:
                            with open(os.path.join(save_dir, f"{mask_dirname}.txt"), 'w') as f:
                                f.write(f"{pred_mask_filepath}\n")
                            masks_intersec[mask_dirname]  += 1
    
    print("Summary. Number of scans with intersections:")
    for key, val in masks_intersec.items():
        print(f"{key}\t{val}")