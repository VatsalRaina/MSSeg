import argparse
import nibabel as nib
import numpy as np
import ants

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--input_flair', type=str, help='Input file location of FLAIR')
parser.add_argument('--input_t1', type=str, help='Input file location of T1')
parser.add_argument('--save_file', type=str, help='Where to save output file of nii.gz')


def main(args):
    
    fi = ants.image_read(args.input_flair)
    mi = ants.image_read(args.input_t1)
    orientation = nib.load(args.input_flair).affine #To save the FLAIR orientation
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'Rigid' ) #RIGID registration
    T1_numpy = mytx['warpedmovout'].numpy()  #Convert the registered image to numpy array
    nifti_out = nib.Nifti1Image(T1_numpy, affine=orientation)
    nifti_out.to_filename(args.save_file) #save registered T1 image


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)