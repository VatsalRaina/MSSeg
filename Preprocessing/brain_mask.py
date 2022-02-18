import argparse
import nibabel as nib
import numpy as np
import ants

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--input_flair', type=str, help='Input file location of FLAIR')
parser.add_argument('--input_brain_mask', type=str, help='Input file location of brain mask')
parser.add_argument('--save_file', type=str, help='Where to save output file of nii.gz')


def main(args):
    
    flair = ants.image_read(args.input_flair).numpy()
    mask = ants.image_read(args.input_t1).numpy()
    orientation = nib.load(args.input_flair).affine #To save the FLAIR orientation

    flair_masked = flair * mask

    nifti_out = nib.Nifti1Image(flair_masked, affine=orientation)
    nifti_out.to_filename(args.save_file) #save masked FLAIR image


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)