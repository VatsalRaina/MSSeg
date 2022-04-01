#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:13:46 2022

@author: nataliia

Implementation of a preprocessing pipeline described below:
    Input: FLAIR and T1
    Steps:
    1. denoise both flair and T1
    2. register T1 to FLAIR
    3. hd-bet T1
    4. brain_mask FLAIR
    5. n4_bis to flair and T1

    Saves flair to output folders

Usage:
    python dataset_preprocessing.py \
    --flair_dir /home/meri/data/msseg_advanced/FLAIR \
    --t1_dir /home/meri/data/msseg_advanced/MPRAGE_T1 \
    --flair_final_dir /home/meri/data/msseg_advanced/FLAIR_preproc \
    --flair_prefix FLAIR.nii.gz \
    --t1_prefix acq-MPRAGE_T1w.nii.gz \
    --root_dir /home/meri/data/msseg_advanced/ \
    --save_tmp
    
python dataset_preprocessing.py \
--flair_dir /home/meri/data/msseg_advanced/FLAIR \
--t1_dir /home/meri/data/msseg_advanced/MPRAGE_T1 \
--flair_final_dir /home/meri/data/msseg_advanced/FLAIR_preproc \
--flair_prefix FLAIR.nii.gz \
--t1_prefix acq-MPRAGE_T1w.nii.gz \
--root_dir /home/meri/data/msseg_advanced/ \
--save_tmp
    
    
--flair_dir [str] path to dir with all FLAIR
--t1_dir [str] similar to T1
--flair_final_dir [str] directory where processed flair images will be saved
--flair_prefix [str] name ending shared across flair files
--t1_prefix [str] name ending shared across t1 files
--root_dir [str] Parent directory for FLAIR and T1. All temporary directories will be created in it
--save_tmp [flag] if sent wont remove tmp dirs
    
"""
import shutil
from dipy.denoise.nlmeans import nlmeans
from dipy.io.image import load_nifti, save_nifti
import argparse
import subprocess
import nibabel as nib
import ants
import SimpleITK as sitk
import os
import pathlib


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--flair_dir', type=str, help='Directory with FLAIR images')
parser.add_argument('--flair_final_dir', type=str, default='FLAIR_processed', 
                    help='Directory to store processed FLAIR images')
parser.add_argument('--t1_dir', type=str, help='Directory with T1 images')
parser.add_argument('--flair_prefix', type=str, help='name ending FLAIR')
parser.add_argument('--t1_prefix', type=str, help='name ending T1')
parser.add_argument('--root_dir', type=str, 
                    help='Parent directory for FLAIR and T1. All temporary directories will be created in it')
parser.add_argument('--save_tmp', action='store_true',
                            help='if sent, does not delete temporary directories')

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
    orientation = nib.load(flair_filename).affine  # To save the FLAIR orientation
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform='Rigid')  # RIGID registration
    T1_numpy = mytx['warpedmovout'].numpy()  # Convert the registered image to numpy array
    nifti_out = nib.Nifti1Image(T1_numpy, affine=orientation)
    nifti_out.to_filename(t1_output_filename)  # save registered T1 image


def brain_masking_toFLAIR(flair_filename, brainmask_filename, output_filename):
    flair = ants.image_read(flair_filename).numpy()
    mask = ants.image_read(brainmask_filename).numpy()
    orientation = nib.load(flair_filename).affine  # To save the FLAIR orientation
    flair_masked = flair * mask
    nifti_out = nib.Nifti1Image(flair_masked, affine=orientation)
    nifti_out.to_filename(output_filename)  # save masked FLAIR image


def bias_correction(input_filepath, output_filepath, shrink_factor=None, image_mask=None, n_iter=None, n_fit=4):
    # TODO: fix the random seed
    inputImage = sitk.ReadImage(input_filepath, sitk.sitkFloat32)
    image = inputImage
    if image_mask is not None:
        maskImage = sitk.ReadImage(image_mask, sitk.sitkUint8)
    else:
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    if shrink_factor is not None:
        image = sitk.Shrink(inputImage, [int(shrink_factor)] * inputImage.GetDimension())
        maskImage = sitk.Shrink(maskImage, [int(shrink_factor)] * inputImage.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    if n_iter is not None:
        corrector.SetMaximumNumberOfIterations([int(n_iter)] * n_fit)
    corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    corrected_image_full_resolution = inputImage / sitk.Exp( log_bias_field )
    sitk.WriteImage(corrected_image, output_filepath)


def preprocessing_pipeline(arg, opt):
    """Apply full preprocessing pipeline to FLAIR images 
    and save them to FLAIR_FINAL/
    """

    def get_brainmask_name(t1_hdbet_filename_):
        name_split = t1_hdbet_filename_.split('.')
        brainmask_name = name_split[0] + '_mask'
        for resid in name_split[1:]:
            brainmask_name += ('.' + resid)
        return brainmask_name

    flair_filepath, t1_filepath = arg
    root_dir = opt.root_dir

    # denoising
    flair_d_filepath = os.path.join(root_dir, "denoised_flair", os.path.basename(flair_filepath))
    t1_d_filepath = os.path.join(root_dir, "denoised_t1", os.path.basename(t1_filepath))
    denoise(input_filename=flair_filepath, output_filename=flair_d_filepath)
    denoise(input_filename=t1_filepath, output_filename=t1_d_filepath)

    # registration
    t1_dr_filepath = os.path.join(root_dir, "registered_t1", os.path.basename(t1_filepath))
    register_T1toFLAIR(flair_filename=flair_d_filepath,
                       t1_filename=t1_d_filepath, t1_output_filename=t1_dr_filepath)

    # skull striping
    t1_hdbet_filepath = os.path.join(root_dir, "hdbet_out", os.path.basename(t1_dr_filepath))
    brainmask_filepath = os.path.join(root_dir, "hdbet_out",
                                      get_brainmask_name(os.path.basename(t1_dr_filepath)))
    res = subprocess.call(
        [
                'hd-bet', '-i', t1_dr_filepath, '-o', t1_hdbet_filepath #, 
#                '-device', 'cpu', '-mode', 'fast', '-tta', '0'
         ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # brain masking
    flair_dm_filepath = os.path.join(root_dir, "masked_flair", os.path.basename(flair_d_filepath))
    brain_masking_toFLAIR(flair_d_filepath, brainmask_filepath, flair_dm_filepath)

    # bias correction
    flair_f_filepath = os.path.join(root_dir, opt.flair_final_dir, os.path.basename(flair_dm_filepath))
    bias_correction(flair_dm_filepath, flair_f_filepath,
                    shrink_factor=None, image_mask=None, n_iter=None, n_fit=4)


"""
Supporting functions
"""

def parse_arguments(opt):
    """Parse all the flair and corresponding filenames from folders
    Returns the list of arguments for `preprocessing_pipeline`
    """
    def check_similarity():
        """Check that actually files matched correctly and return the list of arguments
        """
        assert len(flair_filepaths) == len(
            t1_filepaths), f"Found {len(flair_filepaths)} flair files and {len(t1_filepaths)}"
        args = []
        for p1, p2 in zip(flair_filepaths, t1_filepaths):
            if os.path.basename(p1)[:-len(flair_prefix)] == os.path.basename(p2)[:-len(t1_prefix)]:
                args.append((p1, p2))
            else:
                raise ValueError(f"{p1} and {p2} do not match")
        return args

    flair_dir = opt.flair_dir
    t1_dir = opt.t1_dir
    flair_prefix = opt.flair_prefix
    t1_prefix = opt.t1_prefix

    flair_filepaths = sorted(
        list(pathlib.Path(flair_dir).glob(f'**/*{flair_prefix}'))
    )
    t1_filepaths = sorted(
        list(pathlib.Path(t1_dir).glob(f'**/*{t1_prefix}'))
    )

    return check_similarity()


def create_directories(opt):
    root_dir = opt.root_dir

    for dir_name in TEMP_DIRNAMES + [opt.flair_final_dir]:
        os.makedirs(os.path.join(root_dir, dir_name), exist_ok=True)


def remove_directories(opt):
    root_dir = opt.root_dir

    for dir_name in TEMP_DIRNAMES:
        shutil.rmtree(os.path.join(root_dir, dir_name))


if __name__ == '__main__':
    print("parsing arguments")
    OPT = parser.parse_args()
    print("parsing dataset")
    ARGS = parse_arguments(OPT)
    TEMP_DIRNAMES = ['denoised_flair', 'denoised_t1', 'registered_t1', 
                     'masked_flair', 'hdbet_out']
    print("creating folders")
    create_directories(OPT)
    print("starting processing")
    i_count = 0
    for arg in ARGS:
        preprocessing_pipeline(arg=arg, opt=OPT)
        if i_count % 10 == 0: print(f"processed {i_count} images")
        i_count += 1
    if not OPT.save_tmp:
        print("removing temporary dirs")
        remove_directories(OPT)