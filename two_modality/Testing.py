"""
@author: Francesco La Rosa
"""

import glob
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


def main(temp):

    root_dir= ''  # Path where the trained model is saved
    path_data = ''  # Path where the data is
    flair = sorted(glob(os.path.join(path_data, "*/FLAIR_sk.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))
    mprage = sorted(glob(os.path.join(path_data, "*/MPRAGE_sk.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))
    segs = sorted(glob(os.path.join(path_data, "*/gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))

    N = (len(flair)) # Number of subjects for training/validation, by default using all subjects in the folder
    
    np.random.seed(111)
    indices = np.random.permutation(N)
    # 20 random cases are kept for validation, the others for training
    v=indices[:5]

    test_files=[]
    for j in v:
        test_files = test_files + [{"flair": fl,"mprage": mp,"label": seg} for fl, mp, seg in zip(flair[j:j+1], mprage[j:j+1], segs[j:j+1])]

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
        LoadNiftid(keys=["flair", "mprage", "label"]),
        AddChanneld(keys=["flair", "mprage","label"]),
        Spacingd(keys=["flair", "mprage","label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear","nearest")),
        NormalizeIntensityd(keys=["flair", "mprage"], nonzero=True),
        ConcatItemsd(keys=["flair", "mprage"], name="image"),
        ToTensord(keys=["image", "label"]),
    ]
    )
    #%%

    val_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    
    """
    Select cuda device
    """
    device = torch.device("cuda:0")

    model = UNet(dimensions=3,in_channels=2, out_channels=2,channels=(32, 64, 128, 256, 512),
                  strides=(2, 2, 2, 2),num_res_units=0).to(device)
     

    act = Activations(softmax=True)
    
    subject=0
    model.load_state_dict(torch.load(os.path.join(root_dir, "Best_model_finetuning.pth")))
    
    print()
    print('Running the inference, please wait... ')
    
    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            inputs, gt  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                     batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
            outputs_o = (act(outputs))
            outputs = act(outputs).cpu().numpy()
            outputs = np.squeeze(outputs[0,1])       

            th = 0.2
            outputs[outputs>th]=1
            outputs[outputs<th]=0
            seg= np.squeeze(outputs)
  
            val_labels = gt.cpu().numpy()
            gt = np.squeeze(val_labels)
            value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
            print(value)

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
 
            value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
            
            
            name_patient= os.path.basename(os.path.dirname(test_files[subject]["mprage"]))
            subject+=1
            meta_data = batch_data['mprage_meta_dict']
            for i, data in enumerate(outputs_o):  
                out_meta = {k: meta_data[k][i] for k in meta_data} if meta_data else None
                 

            original_affine = out_meta.get("original_affine", None) if out_meta else None
            affine = out_meta.get("affine", None) if out_meta else None
            spatial_shape = out_meta.get("spatial_shape", None) if out_meta else None
              
            data2=np.copy(seg)
            name = create_file_basename("subject_"+str(name_patient)+".nii.gz","binary_seg",root_dir)
            write_nifti(data2,name,affine=affine,target_affine=original_affine,
                        output_spatial_shape=spatial_shape)        
    
    print()
    print("Inference completed!")
    print("The segmentations have been saved in the following folder: ", os.path.join(root_dir,"binary_seg"))
    print("-------------------------------------------------------------------")
    
#%%
if __name__ == "__main__":
    main(sys.argv)