"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
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
    AddChanneld,AddChannel,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,Spacing,ToTensord,ConcatItemsd,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, AsDiscrete, Activations)
from monai.utils import first, set_determinism
from monai.data import write_nifti, create_file_basename, NiftiDataset
import numpy as np
from scipy import ndimage

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for lesion detection')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_model', type=str, default='', help='Specify the path to the trained model')


# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # THIS MUST BE CHANGED EVERY TIME OR NEEDS TO BE PUT INTO COMMAND LINE ARGUMENT
    
    # MSSEG dev set
    verio = [(1.1, 0.5, 0.5)]
    aera = [(1.25, 1.03, 1.03)]
    ingenia = [(0.7, 0.74, 0.74)]
    resolutions_original = verio + aera + ingenia

    # # MSSEG test set
    # verio = [(1.1, 0.5, 0.5)] * 10
    # discovery = [(0.9, 0.47, 0.47)] * 8
    # aera = [(1.25, 1.03, 1.03)] * 10
    # ingenia = [(0.7, 0.74, 0.74)] * 10
    # resolutions_original = verio + discovery + aera + ingenia
    

    # Choose device
    device = get_default_device()

    root_dir= args.path_model  # Path where the trained model is saved
    path_data = args.path_data  # Path where the data is
    flair = sorted(glob(os.path.join(path_data, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(path_data, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))  


    N = (len(flair)) # Number of subjects for training/validation, by default using all subjects in the folder
    
    # np.random.seed(111)
    # indices = np.random.permutation(N)
    indices = np.arange(N)
    v=indices[:]

    test_files=[]
    for j in v:
        test_files = test_files + [{"image": fl, "label": seg, "label_orig": seg} for fl, seg in zip(flair[j:j+1], segs[j:j+1])]

    print("Testing cases:", len(test_files))
    
    print()
    print("-------------------------------------------------------------------")
    print("Welcome!")
    print()
    print('The MS lesion segmentation will be computed on the following files: ')
    print()
    print(test_files)
    print()
    print("-------------------------------------------------------------------")
    print()
    print('Loading the files and the trained network')
   
    val_transforms = Compose(
    [
        LoadNiftid(keys=["image", "label", "label_orig"]),
        AddChanneld(keys=["image","label", "label_orig"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "label", "label_orig"]),
    ]
    )
    #%%

    val_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    model = UNet(dimensions=3,in_channels=1, out_channels=2,channels=(32, 64, 128, 256, 512),
                  strides=(2, 2, 2, 2),num_res_units=0).to(device)
     

    act = Activations(softmax=True)
    
    subject=0
    model.load_state_dict(torch.load(os.path.join(root_dir, "Best_model_finetuning.pth")))
    
    print()
    print('Running the inference, please wait... ')
    
    th = args.threshold

    model.eval()
    with torch.no_grad():
        all_gts = []
        all_predictions = []
        for batch_data in val_loader:
            inputs, gt, gt_orig  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                    batch_data["label"].type(torch.LongTensor).to(device),#.unsqueeze(0),)
                    batch_data["label_orig"].type(torch.LongTensor).to(device),)
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
            outputs_o = (act(outputs))
            outputs = act(outputs).cpu().numpy()
            outputs = np.squeeze(outputs[0,1])       

            
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
 
            val_labels = gt_orig.cpu().numpy()
            gt = np.squeeze(val_labels)

            all_gts.append(gt)
            all_predictions.append()



    def mapped(seg_unmapped, res):

        val_untransforms = Compose(
        [
            AddChannel(),
            Spacing(pixdim=res, mode="bilinear"),
        ]
        )
            
        return val_untransforms(seg_unmapped)

    metric_sum = 0.0
    metric_count = 0
    for gt, seg_unmapped, res in zip(all_gts, all_predictions, resolutions_original):
        seg = mapped(seg_unmapped, res)

        value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
        metric_count += 1
        metric_sum += value.sum().item()
    metric = metric_sum / metric_count
    print("Dice score:", metric)
            
            

    
#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)