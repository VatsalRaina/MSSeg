"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
         Adapted by Nataliia Molchanova
"""

import argparse
import os
import torch
import sys
import re
from glob import glob
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss
from monai.metrics import compute_meandice, DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,ToTensord,ConcatItemsd,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, AsDiscrete, Activations)
from monai.utils import first, set_determinism
from monai.data import write_nifti, create_file_basename, NiftiDataset
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for lesion detection')
parser.add_argument('--num_models', type=int, default=5, help='Number of models in ensemble')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_model', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--path_save', type=str, default='', help='Specify the path to save the segmentations')
parser.add_argument('--flair_prefix', type=str, default="FLAIR.nii.gz", help='name ending FLAIR')
parser.add_argument('--gts_prefix', type=str, default="gt.nii", help='name ending segmentation mask')
parser.add_argument('--check_dataset', action='store_true',
                    help='if sent, checks that FLAIR and semg masks names correspond to each other')
parser.add_argument('--output_file', type=str, "Specifily the file to store the results")


# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def check_dataset(flair_filepaths, gts_filepaths, args):
    """Check that there are equal amounts of files in both lists and
    the names before prefices are similar for each of the matched pairs of 
    flair image filepath and gts filepath.
    """
    assert len(flair_filepaths) == len(
            gts_filepaths), f"Found {len(flair_filepaths)} flair files and {len(gts_filepaths)}"
    for p1, p2 in zip(flair_filepaths, gts_filepaths):
        if os.path.basename(p1)[:-len(args.flair_prefix)] != os.path.basename(p2)[:-len(args.gts_prefix)]:
            raise ValueError(f"{p1} and {p2} do not match")

def main(args):

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    root_dir= args.path_model  # Path where the trained model is saved
    path_data = args.path_data  # Path where the data is
    flair = sorted(glob(os.path.join(path_data, f"*{args.flair_prefix}")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(args.path_gts, f"*{args.gts_prefix}")),
                  key=lambda i: int(re.sub('\D', '', i)))
    if args.check_dataset:
        check_dataset(flair, segs, args)


    N = (len(flair)) # Number of subjects for training/validation, by default using all subjects in the folder
    
    # np.random.seed(111)
    # indices = np.random.permutation(N)
    indices = np.arange(N)
    v=indices[:]

    test_files=[]
    for j in v:
        test_files = test_files + [{"image": fl, "label": seg} for fl, seg in zip(flair[j:j+1], segs[j:j+1])]

    print("Testing cases:", len(test_files))
    
    print()
    print("-------------------------------------------------------------------")
    print("Welcome!")
    print()
    #print('The MS lesion segmentation will be computed on the following files: ')
    #print()
    #print(test_files)
    print()
    print("-------------------------------------------------------------------")
    print()
    print('Loading the files and the trained network')
   
    val_transforms = Compose(
    [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image","label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "label"]),
    ]
    )
    #%%

    val_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=8)

    K = args.num_models

    models = []
    for i in range(K):
        models.append(UNet(dimensions=3,in_channels=1, out_channels=2,channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),num_res_units=0).to(device))
     

    act = Activations(softmax=True)
    
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(os.path.join(root_dir, "seed" + str(i+1), "Best_model_finetuning.pth")))
        model.eval()

    print()
    print('Running the inference, please wait... ')
    
    th = args.threshold
    r = 0.001

    with torch.no_grad():
        dsc_sum = 0.0
        dsc_norm_sum = 0.0
        fpr_sum = 0.0
        fnr_sum = 0.0
        metric_count = 0
        for count, batch_data in enumerate(val_loader):
            inputs, gt  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                     batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            all_outputs = []
            for model in models:
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
                outputs_o = (act(outputs))
                outputs = act(outputs).cpu().numpy()
                outputs = np.squeeze(outputs[0,1])
                all_outputs.append(outputs)
            all_outputs = np.asarray(all_outputs)
            outputs = np.mean(all_outputs, axis=0)

            
            outputs[outputs>th]=1
            outputs[outputs<th]=0
            seg= np.squeeze(outputs)
  
            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)

            """
            Remove connected components smaller than 10 voxels
            """
            l_min = 9
            labeled_seg, num_labels = ndimage.label(seg)
            label_list = np.unique(labeled_seg)
            num_elements_by_lesion = ndimage.labeled_comprehension(seg,labeled_seg,label_list,np.sum,float, 0)

            seg2 = np.zeros_like(seg)
            for l in range(len(num_elements_by_lesion)):
                if num_elements_by_lesion[l] > l_min:
            # assign voxels to output
                    current_voxels = np.stack(np.where(labeled_seg == l), axis=1)
                    seg2[current_voxels[:, 0],
                        current_voxels[:, 1],
                        current_voxels[:, 2]] = 1
            seg=np.copy(seg2) 

            im_sum = np.sum(seg) + np.sum(gt)
            if im_sum == 0:
                value = 1.0
                dsc_sum += value
                dsc_norm_sum += value
                fpr_sum += value
                fnr_sum += value

            else:

                dsc = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                dsc_sum += dsc.sum().item()
                if np.sum(gt) == 0:
                    k = 1.0
                else:
                    k = (1-r) * np.sum(gt) / ( r * ( len(gt.flatten()) - np.sum(gt) ) )
                tp = np.sum(seg[gt==1])
                fp = np.sum(seg[gt==0])
                fn = np.sum(gt[seg==0])
                fp_scaled = k * fp
                dsc_norm = 2 * tp / (fp_scaled + 2 * tp + fn)
                dsc_norm_sum += dsc_norm

                fpr_sum += fp / ( len(gt.flatten()) - np.sum(gt) )
                if np.sum(gt) == 0:
                    fnr_sum += 1.0
                else:
                    fnr_sum += fn / np.sum(gt)

            metric_count += 1

        with open(args.output_file, 'w') as f:
            dsc = dsc_sum / metric_count
            f.write(f"DSC\t{dsc}\n")
            dsc_norm = dsc_norm_sum / metric_count
            f.write(f"DSCnorm\t{dsc_norm}\n")
            fpr = fpr_sum / metric_count
            f.write(f"FPR\t{fpr}\n")
            fnr = fnr_sum / metric_count
            f.write(f"FNR\t{fnr}\n")
            

#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)