#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:13:46 2022

@author: nataliia

Input: FLAIR and T1, output FLAIR and T1
Steps:
1. denoise both flair and T1
2. register T1 to FLAIR
3. hd-bet T1
4. brain_mask FLAIR
5. n4_bis to flair and T1

Saves T1 and Flair to output folders
"""

from dipy.denoise.nlmeans import nlmeans
from dipy.io.image import load_nifti, save_nifti
import argparse
import subprocess 
import nibabel as nib
import numpy as np
import ants
from __future__ import print_function
import SimpleITK as sitk
import sys
import os
from joblib import Parallel, delayed

#parser = argparse.ArgumentParser(description='Get all command line arguments.')
#parser.add_argument('--input_flair', type=str, help='Input file location of FLAIR')
#parser.add_argument('--input_t1', type=str, help='Input file location of T1')
#parser.add_argument('--save_file', type=str, help='Where to save output file of nii.gz')


"""
Preprocessing functions
"""

def denoise(input_filename, output_filename):
    data, affine = load_nifti(input_filename)
    den = nlmeans(data, sigma=3, patch_radius=3)
    save_nifti(output_filename, den, affine)
    

def register_T1toFLAIR(flair_filename, t1_filename, t1_output_filename):
    fi = ants.image_read(flair_filename)
    mi = ants.image_read(t1_filename)
    orientation = nib.load(flair_filename).affine #To save the FLAIR orientation
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'Rigid' ) #RIGID registration
    T1_numpy = mytx['warpedmovout'].numpy()  #Convert the registered image to numpy array
    nifti_out = nib.Nifti1Image(T1_numpy, affine=orientation)
    nifti_out.to_filename(t1_output_filename) #save registered T1 image
    
    
def hdbet_toT1(t1_filename, output_filename):
    subprocess.call(['hd-bet', '-i', t1_filename, '-o', output_filename])
    
    
def brain_masking_toFLAIR(flair_filename, brainmask_filename, output_filename):
    flair = ants.image_read(flair_filename).numpy()
    mask = ants.image_read(brainmask_filename).numpy()
    orientation = nib.load(flair_filename).affine #To save the FLAIR orientation
    flair_masked = flair * mask
    nifti_out = nib.Nifti1Image(flair_masked, affine=orientation)
    nifti_out.to_filename(output_filename) #save masked FLAIR image
    
def bias_correction(input_filepath, output_filepath, shrink_factor=None, image_mask=None, n_iter=None, n_fit=4):
    inputImage = sitk.ReadImage(input_filepath, sitk.sitkFloat32)
    image = inputImage
    if image_mask is not None:
        maskImage = sitk.ReadImage(image_mask, sitk.sitkUint8)
    else:
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    if shrink_factor is not None:
        image = sitk.Shrink(inputImage,[int(shrink_factor)] * inputImage.GetDimension())
        maskImage = sitk.Shrink(maskImage,[int(shrink_factor)] * inputImage.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    if n_iter is not None:
        corrector.SetMaximumNumberOfIterations([int(n_iter)] * n_fit)
    corrected_image = corrector.Execute(image, maskImage)
#    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
#    corrected_image_full_resolution = inputImage / sitk.Exp( log_bias_field ) 
    sitk.WriteImage(corrected_image, output_filepath)
       
def preprocessing_pipeline1(args):
    """Denoising and registration.
    Results are saved to denoised_(flait|t1)/ and registered_t1/
    Returns the filenames 
    """
    flair_filename, t1_filename, root_dir = args
    
    # saves to the denoise_flair and denoise_T1 directories
    # TODO: make sure that these directories exist before the loop
    flair_d_filename = os.path.join(root_dir, "denoised_flair", os.path.basename(flair_filename))
    t1_d_filename = os.path.join(root_dir, "denoised_t1", os.path.basename(t1_filename))
    denoise(flair_filename, flair_d_filename)
    denoise(t1_filename, t1_d_filename)
    
    # registration
    t1_dr_filename = os.path.join(root_dir, "registered_t1", os.path.basename(t1_filename))
    register_T1toFLAIR(flair_d_filename, t1_d_filename, t1_dr_filename)
    
    
def preprocessing_pipeline2(args):
    """Apply brain masking and skull striping to flair 
    and save it to masked_flair/ and FLAIR_final/.
    """
    flair_d_filename, brainmask_filename, root_dir = args
    # TODO: make sure that these directories exist before the loop
    
    # brain masking
    flair_dm_filename = os.path.join(root_dir, "masked_flair", os.path.basename(flair_d_filename))
    brain_masking_toFLAIR(flair_d_filename, brainmask_filename, flair_dm_filename)
    
    # bias correction
    flair_f_filename = os.path.join(root_dir, "FLAIR_final", os.path.basename(flair_d_filename))
    bias_correction(flair_dm_filename, flair_f_filename, 
                    shrink_factor=None, image_mask=None, n_iter=None, n_fit=4)
    

def preprocessing_pipeline(args, opt):
    """Apply full preprocessing pipeline to FLAIR images 
    and save them to FLAIR_FINAL/
    """
    args2 = Parallel(n_jobs=opt.n_jobs)(delayed(preprocessing_pipeline1)(i) for arg in args)
    
    
    