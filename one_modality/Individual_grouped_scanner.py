"""
Plot performance of each model against lesion load (percentage lesion voxels) of patients.
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
import pandas as pd

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for lesion detection')
parser.add_argument('--num_models', type=int, default=5, help='Number of models in ensemble')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_data2', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_gts2', type=str, default='', help='Specify the path to the test gts directory')
parser.add_argument('--path_model1', type=str, default='', help='Specify the dir to all the trained models')
parser.add_argument('--path_model2', type=str, default='', help='Specify the dir to all the trained models')
parser.add_argument('--path_model3', type=str, default='', help='Specify the dir to all the trained models')
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

    path_data = args.path_data  # Path where the data is
    flair = sorted(glob(os.path.join(path_data, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(args.path_gts, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))        


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

    K = args.num_models

    models1 = []
    for i in range(K):
        models1.append(UNet(dimensions=3,in_channels=1, out_channels=2,channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),num_res_units=0).to(device))

     
    models2 = []
    for i in range(K):
        models2.append(UNet(dimensions=3,in_channels=1, out_channels=2,channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),num_res_units=0).to(device))

    models3 = []
    for i in range(K):
        models3.append(UNet(dimensions=3,in_channels=1, out_channels=2,channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),num_res_units=0).to(device))

    act = Activations(softmax=True)
    
    for i, model in enumerate(models1):
        model.load_state_dict(torch.load(args.path_model1 + "seed" + str(i+1) + "/Best_model_finetuning.pth"))
        model.eval()

    for i, model in enumerate(models2):
        model.load_state_dict(torch.load(args.path_model2 + "seed" + str(i+1) + "/Best_model_finetuning.pth"))
        model.eval()

    for i, model in enumerate(models3):
        model.load_state_dict(torch.load(args.path_model3 + "seed" + str(i+1) + "/Best_model_finetuning.pth"))
        model.eval()

    print()
    print('Running the inference, please wait... ')
    
    th = args.threshold

    verio_patients = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    discovery_patients = [11, 12, 13, 14, 15, 16, 17, 18]
    aera_patients = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    ingenia_patients = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

    verio_dsc_1 = []
    discovery_dsc_1 = []
    aera_dsc_1 = []
    ingenia_dsc_1 = []

    verio_dsc_norm_1 = []
    discovery_dsc_norm_1 = []
    aera_dsc_norm_1 = []
    ingenia_dsc_norm_1 = []


    verio_dsc_2 = []
    discovery_dsc_2 = []
    aera_dsc_2 = []
    ingenia_dsc_2 = []

    verio_dsc_norm_2 = []
    discovery_dsc_norm_2 = []
    aera_dsc_norm_2 = []
    ingenia_dsc_norm_2 = []


    verio_dsc_3 = []
    discovery_dsc_3 = []
    aera_dsc_3 = []
    ingenia_dsc_3 = []

    verio_dsc_norm_3 = []
    discovery_dsc_norm_3 = []
    aera_dsc_norm_3 = []
    ingenia_dsc_norm_3 = []

    r = 0.001

    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            patient_num = count + 1

            inputs, gt  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                     batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            all_outputs = []
            for model in models1:
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
            else:
                value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                value = value.sum().item()

            if patient_num in verio_patients:
                verio_dsc_1.append(value)
            if patient_num in discovery_patients:
                discovery_dsc_1.append(value)
            if patient_num in aera_patients:
                aera_dsc_1.append(value)
            if patient_num in ingenia_patients:
                ingenia_dsc_1.append(value)

            print(patient_num)

            # Calculate the normalised DSC score now

            im_sum = np.sum(seg) + np.sum(gt)
            if im_sum == 0:
                value = 1.0
            else:
                k = (1-r) * np.sum(gt) / ( r * ( len(gt.flatten()) - np.sum(gt) ) )
                tp = np.sum(seg[gt==1])
                fp = np.sum(seg[gt==0])
                fn = np.sum(gt[seg==0])
                fp_scaled = k * fp
                value = 2 * tp / (fp_scaled + 2 * tp + fn)

            if patient_num in verio_patients:
                verio_dsc_norm_1.append(value)
            if patient_num in discovery_patients:
                discovery_dsc_norm_1.append(value)
            if patient_num in aera_patients:
                aera_dsc_norm_1.append(value)
            if patient_num in ingenia_patients:
                ingenia_dsc_norm_1.append(value)


            all_outputs = []
            for model in models2:
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
            else:
                value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                value = value.sum().item()

            if patient_num in verio_patients:
                verio_dsc_2.append(value)
            if patient_num in discovery_patients:
                discovery_dsc_2.append(value)
            if patient_num in aera_patients:
                aera_dsc_2.append(value)
            if patient_num in ingenia_patients:
                ingenia_dsc_2.append(value)

            # Calculate the normalised DSC score now

            im_sum = np.sum(seg) + np.sum(gt)
            if im_sum == 0:
                value = 1.0
            else:
                k = (1-r) * np.sum(gt) / ( r * ( len(gt.flatten()) - np.sum(gt) ) )
                tp = np.sum(seg[gt==1])
                fp = np.sum(seg[gt==0])
                fn = np.sum(gt[seg==0])
                fp_scaled = k * fp
                value = 2 * tp / (fp_scaled + 2 * tp + fn)

            if patient_num in verio_patients:
                verio_dsc_norm_2.append(value)
            if patient_num in discovery_patients:
                discovery_dsc_norm_2.append(value)
            if patient_num in aera_patients:
                aera_dsc_norm_2.append(value)
            if patient_num in ingenia_patients:
                ingenia_dsc_norm_2.append(value)



            all_outputs = []
            for model in models3:
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
            else:
                value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                value = value.sum().item()

            if patient_num in verio_patients:
                verio_dsc_3.append(value)
            if patient_num in discovery_patients:
                discovery_dsc_3.append(value)
            if patient_num in aera_patients:
                aera_dsc_3.append(value)
            if patient_num in ingenia_patients:
                ingenia_dsc_3.append(value)

            # Calculate the normalised DSC score now

            im_sum = np.sum(seg) + np.sum(gt)
            if im_sum == 0:
                value = 1.0
            else:
                k = (1-r) * np.sum(gt) / ( r * ( len(gt.flatten()) - np.sum(gt) ) )
                tp = np.sum(seg[gt==1])
                fp = np.sum(seg[gt==0])
                fn = np.sum(gt[seg==0])
                fp_scaled = k * fp
                value = 2 * tp / (fp_scaled + 2 * tp + fn)

            if patient_num in verio_patients:
                verio_dsc_norm_3.append(value)
            if patient_num in discovery_patients:
                discovery_dsc_norm_3.append(value)
            if patient_num in aera_patients:
                aera_dsc_norm_3.append(value)
            if patient_num in ingenia_patients:
                ingenia_dsc_norm_3.append(value)



    path_data = args.path_data2  # Path where the data is
    flair = sorted(glob(os.path.join(path_data, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(args.path_gts2, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))        


    N = (len(flair)) # Number of subjects for training/validation, by default using all subjects in the folder
    
    indices = np.arange(N)
    v=indices[:]

    test_files=[]
    for j in v:
        test_files = test_files + [{"image": fl, "label": seg} for fl, seg in zip(flair[j:j+1], segs[j:j+1])]

    print("Testing cases:", len(test_files))

    val_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    magnetomTrio_dsc_1 = []
    magnetomTrio_dsc_norm_1 = []

    magnetomTrio_dsc_2 = []
    magnetomTrio_dsc_norm_2 = []

    magnetomTrio_dsc_3 = []
    magnetomTrio_dsc_norm_3 = []

    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            patient_num = count + 1

            inputs, gt  = (
                    batch_data["image"].to(device),#.unsqueeze(0),
                     batch_data["label"].type(torch.LongTensor).to(device),)#.unsqueeze(0),)
            roi_size = (96, 96, 96)
            sw_batch_size = 4

            all_outputs = []
            for model in models1:
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
            else:
                value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                value = value.sum().item()

            magnetomTrio_dsc_1.append(value)
            print(patient_num)

            # Calculate the normalised DSC score now

            im_sum = np.sum(seg) + np.sum(gt)
            if im_sum == 0:
                value = 1.0
            else:
                k = (1-r) * np.sum(gt) / ( r * ( len(gt.flatten()) - np.sum(gt) ) )
                tp = np.sum(seg[gt==1])
                fp = np.sum(seg[gt==0])
                fn = np.sum(gt[seg==0])
                fp_scaled = k * fp
                value = 2 * tp / (fp_scaled + 2 * tp + fn)

            magnetomTrio_dsc_norm_1.append(value)


            all_outputs = []
            for model in models2:
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
            else:
                value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                value = value.sum().item()

            magnetomTrio_dsc_2.append(value)

            # Calculate the normalised DSC score now

            im_sum = np.sum(seg) + np.sum(gt)
            if im_sum == 0:
                value = 1.0
            else:
                k = (1-r) * np.sum(gt) / ( r * ( len(gt.flatten()) - np.sum(gt) ) )
                tp = np.sum(seg[gt==1])
                fp = np.sum(seg[gt==0])
                fn = np.sum(gt[seg==0])
                fp_scaled = k * fp
                value = 2 * tp / (fp_scaled + 2 * tp + fn)

            magnetomTrio_dsc_norm_2.append(value)


            all_outputs = []
            for model in models3:
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
            else:
                value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))
                value = value.sum().item()

            magnetomTrio_dsc_3.append(value)
            
            # Calculate the normalised DSC score now

            im_sum = np.sum(seg) + np.sum(gt)
            if im_sum == 0:
                value = 1.0
            else:
                k = (1-r) * np.sum(gt) / ( r * ( len(gt.flatten()) - np.sum(gt) ) )
                tp = np.sum(seg[gt==1])
                fp = np.sum(seg[gt==0])
                fn = np.sum(gt[seg==0])
                fp_scaled = k * fp
                value = 2 * tp / (fp_scaled + 2 * tp + fn)

            magnetomTrio_dsc_norm_3.append(value)



    # data = []
    # for dsc in verio_dsc:
    #     data.append({"Scanner": "Verio", "DSC": dsc})
    # for dsc in discovery_dsc:
    #     data.append({"Scanner": "Discovery", "DSC": dsc})
    # for dsc in aera_dsc:
    #     data.append({"Scanner": "Aera", "DSC": dsc})
    # for dsc in ingenia_dsc:
    #     data.append({"Scanner": "Ingenia", "DSC": dsc})
    # for dsc in magnetomTrio_dsc:
    #     data.append({"Scanner": "Magnetom Trio", "DSC": dsc})
    # df = pd.DataFrame(data)
    
    # ax = sns.boxplot(x="Scanner", y="DSC", data=df)
    # ax = sns.swarmplot(x="Scanner", y="DSC", data=df, color=".25")
    # plt.savefig('scanner_all.png')
    # plt.clf()

    data = []

    for dsc in verio_dsc_norm_1:
        data.append({"Model": "MSSEG", "Scanner": "Verio", r"$\overline{DSC}$": dsc})
    for dsc in discovery_dsc_norm_1:
        data.append({"Model": "MSSEG", "Scanner": "Discovery", r"$\overline{DSC}$": dsc})
    for dsc in aera_dsc_norm_1:
        data.append({"Model": "MSSEG", "Scanner": "Aera", r"$\overline{DSC}$": dsc})
    for dsc in ingenia_dsc_norm_1:
        data.append({"Model": "MSSEG", "Scanner": "Ingenia", r"$\overline{DSC}$": dsc})
    for dsc in magnetomTrio_dsc_norm_1:
        data.append({"Model": "MSSEG", "Scanner": "Magnetom", r"$\overline{DSC}$": dsc})

    for dsc in verio_dsc_norm_2:
        data.append({"Model": "PubMRI", "Scanner": "Verio", r"$\overline{DSC}$": dsc})
    for dsc in discovery_dsc_norm_2:
        data.append({"Model": "PubMRI", "Scanner": "Discovery", r"$\overline{DSC}$": dsc})
    for dsc in aera_dsc_norm_2:
        data.append({"Model": "PubMRI", "Scanner": "Aera", r"$\overline{DSC}$": dsc})
    for dsc in ingenia_dsc_norm_2:
        data.append({"Model": "PubMRI", "Scanner": "Ingenia", r"$\overline{DSC}$": dsc})
    for dsc in magnetomTrio_dsc_norm_2:
        data.append({"Model": "PubMRI", "Scanner": "Magnetom", r"$\overline{DSC}$": dsc})

    for dsc in verio_dsc_norm_3:
        data.append({"Model": "Combined", "Scanner": "Verio", r"$\overline{DSC}$": dsc})
    for dsc in discovery_dsc_norm_3:
        data.append({"Model": "Combined", "Scanner": "Discovery", r"$\overline{DSC}$": dsc})
    for dsc in aera_dsc_norm_3:
        data.append({"Model": "Combined", "Scanner": "Aera", r"$\overline{DSC}$": dsc})
    for dsc in ingenia_dsc_norm_3:
        data.append({"Model": "Combined", "Scanner": "Ingenia", r"$\overline{DSC}$": dsc})
    for dsc in magnetomTrio_dsc_norm_3:
        data.append({"Model": "Combined", "Scanner": "Magnetom", r"$\overline{DSC}$": dsc})

    df = pd.DataFrame(data)
    
    ax = sns.boxplot(x="Scanner", y=r"$\overline{DSC}$", hue="Model", data=df)
    ax = sns.swarmplot(x="Scanner", y=r"$\overline{DSC}$", hue="Model", data=df, color=".25")
    plt.savefig('scanner_all.png')
    plt.clf()


#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)