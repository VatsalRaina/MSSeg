"""
Save .nii images as equivalent .mha files (necessary for GranChallenge)
"""
import argparse
import SimpleITK as sitk

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--in_file', type=str, default=None, help='.nii file input')
parser.add_argument('--out_file', type=str, default=None, help='.mha file to be produced')


def main(args):

    reader = sitk.ImageFileReader()
    #reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(args.in_file)
    image = reader.Execute()

    writer = sitk.ImageFileWriter()
    writer.SetFileName(args.out_file)
    writer.Execute(image)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
