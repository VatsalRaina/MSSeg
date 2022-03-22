"""
Register the ground-truth to the raw FLAIR space from the ISBI preprocessed FLAIR.
"""

import argparse
import nibabel as nib
import numpy as np
import ants

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--input_flair_proc', type=str, help='Input file location of FLAIR ISBI preprocessed')
parser.add_argument('--input_flair_raw', type=str, help='Input file location of FLAIR raw')
parser.add_argument('--input_gt', type=str, help='Input file location of gt in FLAIR ISBI preprocessed space')
parser.add_argument('--save_file_temp', type=str, help='Where to save temp output file of nii.gz')
parser.add_argument('--save_file', type=str, help='Where to save output file of nii')


def main(args):
    
    fi = ants.image_read(args.input_flair_proc)
    mi = ants.image_read(args.input_flair_raw)
    orientation = nib.load(args.input_flair_raw).affine #Save the raw FLAIR orientation
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'Rigid' ) #RIGID registration of processed FLAIR onto raw

    fl_numpy = mytx['warpedmovout'].numpy()  #Convert the registered image to numpy array
    nifti_out = nib.Nifti1Image(fl_numpy, affine=orientation)
    nifti_out.to_filename(args.save_file_temp)

    gt_proc = ants.image_read(args.input_gt)
    # Apply rigid transform to gt
    gt_raw = ants.apply_transforms(fixed=fi, moving=gt_proc, transformlist=mytx['fwdtransforms'], interpolator="nearestNeighbor")
    
    gt = gt_raw.numpy()
    nifti_out = nib.Nifti1Image(gt, affine=orientation)
    nifti_out.to_filename(args.save_file) #save registered gt image


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)