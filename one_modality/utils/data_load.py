import os
import re
from glob import glob
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AddChanneld, Compose, CropForegroundd, LoadNiftid, Orientationd, RandCropByPosNegLabeld,
    ScaleIntensityRanged, Spacingd, ToTensord, ConcatItemsd, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd, AsDiscrete, Activations)
from scipy import ndimage
import numpy as np


def check_dataset(flair_filepaths, gts_filepaths, args):
    """Check that there are equal amounts of files in both lists and
    the names before prefices are similar for each of the matched pairs of
    flair image filepath and gts filepath.
    Parameters:
        - flair_filepaths (list) - list of paths to flair images
        - gts_filepaths (list) - list of paths to lesion masks of flair images
        - args (argparse.ArgumentParser) - system arguments, should contain `flair_prefix` and `gts_prefix`
    """
    assert len(flair_filepaths) == len(
        gts_filepaths), f"Found {len(flair_filepaths)} flair files and {len(gts_filepaths)}"
    for p1, p2 in zip(flair_filepaths, gts_filepaths):
        if os.path.basename(p1)[:-len(args.flair_prefix)] != os.path.basename(p2)[:-len(args.gts_prefix)]:
            raise ValueError(f"{p1} and {p2} do not match")


def get_data_loader(args):
    """ Create a torch DataLoader based on the system arguments.
    Required options in the `args`: path_data, flair_frefix, path_gts, gts_prefix, check_dataset, num_workers

    """
    flair = sorted(glob(os.path.join(args.path_data, f"*{args.flair_prefix}")),
                   key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(args.path_gts, f"*{args.gts_prefix}")),
                  key=lambda i: int(re.sub('\D', '', i)))
    if args.check_dataset:
        check_dataset(flair, segs, args)

    print(f"Initializing the dataset. Number of subjects {len(flair)}")

    test_files = [{"image": fl, "label": seg} for fl, seg in zip(flair, segs)]

    val_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=0.5, num_workers=args.num_workers)
    return DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)


def remove_connected_components(segmentation, l_min=9):
    """
    Remove small lesions
    """
    labeled_seg, num_labels = ndimage.label(segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        if n_el > l_min:
            current_voxels = np.stack(np.where(labeled_seg == i_el), axis=1)
            seg2[current_voxels[:, 0],
                 current_voxels[:, 1],
                 current_voxels[:, 2]] = 1
    return seg2

def generate_pred_filename(gt_filename):
    """Create the file name of the prediction from the name of the segmentation
    in a form: <gt_filename_base>_pred.nii.gz
    """
    return gt_filename.split('.')[0] + '_pred.nii.gz'