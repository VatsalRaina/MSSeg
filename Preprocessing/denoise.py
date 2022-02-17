from scipy import load
from dipy.denoise.nlmeans import nlmeans
from dipy.io.image import load_nifti, save_nifti
import argparse

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--input_file', type=str, help='Input file location of nii.gz')
parser.add_argument('--save_file', type=str, help='Where to save output file of nii.gz')


def main(args):
    
    data, affine = load_nifti(args.input_file)
    den = nlmeans(data, patch_radius=3)
    save_nifti(args.save_file, den, affine)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)