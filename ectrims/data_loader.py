from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss
from monai.metrics import compute_meandice
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,ToTensord,ConcatItemsd,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, RandScaleIntensityd, Activations,SqueezeDimd)
import numpy as np
import os
from glob import glob
import re

# train_transforms = Compose(
#     [
#         LoadNiftid(keys=["flair", "mp2rage" ,"label"]),

#         # SqueezeDimd(keys=["flair", "mp2rage"], dim=-1),

#         AddChanneld(keys=["flair", "mp2rage" ,"label"]),
#         Spacingd(keys=["flair" ,"mp2rage" ,"label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear" ,"nearest")),
#         NormalizeIntensityd(keys=["flair", "mp2rage"], nonzero=True),
#         RandShiftIntensityd(keys="flair" ,offsets=0.1 ,prob=1.0),
#         RandShiftIntensityd(keys="mp2rage" ,offsets=0.1 ,prob=1.0),
#         RandScaleIntensityd(keys="flair" ,factors=0.1 ,prob=1.0),
#         RandScaleIntensityd(keys="mp2rage" ,factors=0.1 ,prob=1.0),
#         ConcatItemsd(keys=["flair", "mp2rage"], name="image"),
#         RandCropByPosNegLabeld(keys=["image", "label"] ,label_key="label" ,spatial_size=(128, 128, 128),
#                                pos=4 ,neg=1 ,num_samples=32 ,image_key="image"),
#         RandSpatialCropd(keys=["image", "label"], roi_size=(96, 96, 96), random_center=True, random_size=False),
#         RandFlipd (keys=["image", "label"] ,prob=0.5 ,spatial_axis=(0 ,1 ,2)),
#         RandRotate90d (keys=["image", "label"] ,prob=0.5 ,spatial_axes=(0 ,1)),
#         RandRotate90d (keys=["image", "label"] ,prob=0.5 ,spatial_axes=(1 ,2)),
#         RandRotate90d (keys=["image", "label"] ,prob=0.5 ,spatial_axes=(0 ,2)),
#         RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
#                     rotate_range=(np.pi /12, np.pi /12, np.pi /12), scale_range=(0.1, 0.1, 0.1)
#                     ,padding_mode='border'),
#         ToTensord(keys=["image", "label"]),
#     ]
# )
# val_transforms = Compose(
#     [
#         LoadNiftid(keys=["flair", "mp2rage", "label"]),
#         # SqueezeDimd(keys=["flair", "mp2rage"], dim=-1),
#         AddChanneld(keys=["flair", "mp2rage" ,"label"]),
#         Spacingd(keys=["flair", "mp2rage" ,"label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear" ,"nearest")),
#         NormalizeIntensityd(keys=["flair", "mp2rage"], nonzero=True),
#         ConcatItemsd(keys=["flair", "mp2rage"], name="image"),
#         ToTensord(keys=["image", "label"]),
#     ]
# )

train_transforms = Compose(
    [
        LoadNiftid(keys=["flair", "mp2rage" ,"label", 'cl_mask']),

        # SqueezeDimd(keys=["flair", "mp2rage"], dim=-1),

        AddChanneld(keys=["flair", "mp2rage" ,"label", 'cl_mask']),
        Spacingd(keys=["flair" ,"mp2rage" ,"label", 'cl_mask'], pixdim=(1.0, 1.0, 1.0), 
                 mode=("bilinear", "bilinear" ,"nearest", "nearest")),
        NormalizeIntensityd(keys=["flair", "mp2rage"], nonzero=True),
        RandShiftIntensityd(keys="flair" ,offsets=0.1 ,prob=1.0),
        RandShiftIntensityd(keys="mp2rage" ,offsets=0.1 ,prob=1.0),
        RandScaleIntensityd(keys="flair" ,factors=0.1 ,prob=1.0),
        RandScaleIntensityd(keys="mp2rage" ,factors=0.1 ,prob=1.0),
        ConcatItemsd(keys=["flair", "mp2rage"], name="image"),
        RandCropByPosNegLabeld(keys=["image", "label", 'cl_mask'] ,label_key="label",
                               spatial_size=(128, 128, 128),
                               pos=4 ,neg=1 ,num_samples=32),
        RandSpatialCropd(keys=["image", "label", 'cl_mask'], roi_size=(96, 96, 96), 
                         random_center=True, random_size=False),
        RandFlipd (keys=["image", "label", 'cl_mask'] ,prob=0.5 ,spatial_axis=(0 ,1 ,2)),
        RandRotate90d (keys=["image", "label", 'cl_mask'] ,prob=0.5 ,spatial_axes=(0 ,1)),
        RandRotate90d (keys=["image", "label", 'cl_mask'] ,prob=0.5 ,spatial_axes=(1 ,2)),
        RandRotate90d (keys=["image", "label", 'cl_mask'] ,prob=0.5 ,spatial_axes=(0 ,2)),
        RandAffined(keys=['image', 'label', 'cl_mask'], 
                    mode=('bilinear', 'nearest', 'nearest'), 
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(np.pi /12, np.pi /12, np.pi /12), 
                    scale_range=(0.1, 0.1, 0.1)
                    ,padding_mode='border'),
        ToTensord(keys=["image", "label", 'cl_mask']),
    ]
)
val_transforms = Compose(
    [
        LoadNiftid(keys=["flair", "mp2rage", "label", 'cl_mask']),
        # SqueezeDimd(keys=["flair", "mp2rage"], dim=-1),
        AddChanneld(keys=["flair", "mp2rage" ,"label", 'cl_mask']),
        Spacingd(keys=["flair", "mp2rage" ,"label", 'cl_mask'], 
                 pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear" ,"nearest", "nearest")),
        NormalizeIntensityd(keys=["flair", "mp2rage"], nonzero=True),
        ConcatItemsd(keys=["flair", "mp2rage"], name="image"),
        ToTensord(keys=["image", "label", 'cl_mask']),
    ]
)
val_gt_transforms = Compose(
    [
        LoadNiftid(keys=["label"]),
        AddChanneld(keys=["label"]),
        Spacingd(keys=["label"], pixdim=(1.0, 1.0, 1.0), mode=("nearest")),
        ToTensord(keys=["label"]),
    ]
)


def check_dataset(filepaths_list, prefixes):
    """Check that there are equal amounts of files in both lists and
    the names before prefices are similar for each of the matched pairs of
    flair image filepath and gts filepath.
    Parameters:
        - flair_filepaths (list) - list of paths to flair images
        - gts_filepaths (list) - list of paths to lesion masks of flair images
        - args (argparse.ArgumentParser) - system arguments, should contain `flair_prefix` and `gts_prefix`
    """
    if len(filepaths_list) > 1:
        filepaths_ref = filepaths_list[0]
        for filepaths in filepaths_list[1:]:
            assert len(filepaths_ref) == len(
                filepaths), f"Found {len(filepaths_ref)} ref files and {len(filepaths)}"
        for sub_filepaths in list(map(list, zip(*filepaths_list))):
            filepath_ref = sub_filepaths[0]
            prefix_ref = prefixes[0]
            for filepath, prefix in zip(sub_filepaths[1:], prefixes[1:]):
                if os.path.basename(filepath_ref)[:-len(prefix_ref)] != os.path.basename(filepath)[:-len(prefix)]:
                    raise ValueError(f"{filepath_ref} and {filepath} do not match")


def get_data_loader(path_flair, path_mp2rage, path_gts, flair_prefix, mp2rage_prefix, gts_prefix, check=True,
                    transforms=None, num_workers=0, batch_size=1, cache_rate=0.5):
    """ Create a torch DataLoader based on the system arguments.
    """
    flair = sorted(glob(os.path.join(path_flair, f"*{flair_prefix}")),
                   key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    mp2rage = sorted(glob(os.path.join(path_mp2rage, f"*{mp2rage_prefix}")),
                   key=lambda i: int(re.sub('\D', '', i)))
    segs = sorted(glob(os.path.join(path_gts, f"*{gts_prefix}")),
                  key=lambda i: int(re.sub('\D', '', i)))
    if check:
        check_dataset([flair, mp2rage, segs], [flair_prefix, mp2rage_prefix, gts_prefix])

    print(f"Initializing the dataset. Number of subjects {len(flair)}")

    files = [{"flair": fl, "mp2rage": mp2, "label": seg} for fl, mp2, seg in zip(flair, mp2rage, segs)]

    val_ds = CacheDataset(data=files, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(val_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)

def get_data_loader_with_cl(path_flair, path_mp2rage, path_gts, flair_prefix, mp2rage_prefix, gts_prefix, check=True,
                    transforms=None, num_workers=0, batch_size=1, cache_rate=0.5
                    ):
    """ Create a torch DataLoader based on the system arguments.
    """
    flair = sorted(glob(os.path.join(path_flair, f"*{flair_prefix}")),
                   key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    mp2rage = sorted(glob(os.path.join(path_mp2rage, f"*{mp2rage_prefix}")),
                   key=lambda i: int(re.sub('\D', '', i)))
    segs = sorted(glob(os.path.join(path_gts, f"*{gts_prefix}")),
                  key=lambda i: int(re.sub('\D', '', i)))
    segs_cl = sorted(glob(os.path.join(path_gts, "*cortical_lesions.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i)))
    
    if check:
        check_dataset([flair, mp2rage, segs, segs_cl], 
                      [flair_prefix, mp2rage_prefix, gts_prefix, 'cortical_lesions.nii.gz'])

    print(f"Initializing the dataset. Number of subjects {len(flair)}")

    files = [{"flair": fl, "mp2rage": mp2, "label": seg, "cl_mask": seg_cl} 
             for fl, mp2, seg, seg_cl in zip(flair, mp2rage, segs, segs_cl)]

    val_ds = CacheDataset(data=files, transform=transforms, 
                          cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(val_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)


def get_data_loader_gt(path_gts, gts_prefix, transforms=None, num_workers=0, batch_size=1):
    """ Create a torch DataLoader based on the system arguments.
    """
    segs = sorted(glob(os.path.join(path_gts, f"*{gts_prefix}")),
                  key=lambda i: int(re.sub('\D', '', i)))
    
    print(f"Initializing the dataset. Number of subjects {len(segs)}")

    files = [{"label": seg} for seg in segs]

    val_ds = CacheDataset(data=files, transform=transforms, cache_rate=0.3, num_workers=num_workers)
    return DataLoader(val_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)
