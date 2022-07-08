"""
Save .npy images as equivalent .mha files
"""
import argparse
import SimpleITK as sitk
import numpy as np

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--in_file', type=str, default=None, help='.npy file input')
parser.add_argument('--out_file', type=str, default=None, help='.mha file to be produced')


def main(args):

    img_array = np.load(args.in_file)
    image = sitk.GetImageFromArray(img_array)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(args.out_file)
    writer.Execute(image)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)