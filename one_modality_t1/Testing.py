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
import itertools

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for lesion detection')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_model', type=str, default='', help='Specify the path to the trained model')
parser.add_argument('--path_save', type=str, default='', help='Specify the path to save the segmentations')



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
    

    # Choose device
    device = get_default_device()

    root_dir= args.path_model  # Path where the trained model is saved
    path_data = args.path_data  # Path where the data is
    t1 = sorted(glob(os.path.join(path_data, "*T1.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(args.path_gts, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))  


    N = (len(t1)) # Number of subjects for training/validation, by default using all subjects in the folder
    
    # np.random.seed(111)
    # indices = np.random.permutation(N)
    indices = np.arange(N)
    v=indices[:]

    test_files=[]
    for j in v:
        test_files = test_files + [{"image": t, "label": seg} for t, seg in zip(t1[j:j+1], segs[j:j+1])]

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
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image","label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "label"]),
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
    metric_sum = 0.0
    metric_count = 0
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            inputs, gt = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                    batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
            outputs_o = (act(outputs))
            outputs = act(outputs).cpu().numpy()
            outputs = np.squeeze(outputs[0,1])       

            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)

            outputs[outputs>th]=1
            outputs[outputs<th]=0
            seg= np.squeeze(outputs)   

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
                metric_sum += value
            else:
                value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                metric_sum += value.sum().item()
            metric_count += 1

            # # Save as predictions as nii file here in original space

            # meta_data = batch_data['image_meta_dict']
            # for i, data in enumerate(outputs_o):  
            #     out_meta = {k: meta_data[k][i] for k in meta_data} if meta_data else None

            # original_affine = out_meta.get("original_affine", None) if out_meta else None
            # affine = out_meta.get("affine", None) if out_meta else None
            # spatial_shape = out_meta.get("spatial_shape", None) if out_meta else None
              
            # data2=np.copy(seg)
            # name = args.path_save+str(count+1)+".nii.gz"
            # write_nifti(data2,name,affine=affine,target_affine=original_affine,
            #             output_spatial_shape=spatial_shape)        

        metric = metric_sum / metric_count
        print("Dice score:", metric)
            
            

    
#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)