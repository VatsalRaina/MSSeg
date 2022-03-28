"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from glob import glob
import torch
import re
from torch import nn
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss
from monai.metrics import compute_meandice
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,Compose,CropForegroundd,LoadNiftid,Orientationd,RandCropByPosNegLabeld,
    ScaleIntensityRanged,Spacingd,ToTensord,ConcatItemsd,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, RandScaleIntensityd, Activations,SqueezeDimd)
from monai.utils import first
import numpy as np
import random

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify the initial learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--path_train_data1', type=str, default='', help='Specify the path to the training data files directory')
parser.add_argument('--path_train_gts1', type=str, default='', help='Specify the path to the training gts files directory')
parser.add_argument('--path_val_data1', type=str, default='', help='Specify the path to the validation data files directory')
parser.add_argument('--path_val_gts1', type=str, default='', help='Specify the path to the validation gts files directory')
parser.add_argument('--path_train_data2', type=str, default='', help='Specify the path to the training data files directory')
parser.add_argument('--path_train_gts2', type=str, default='', help='Specify the path to the training gts files directory')
parser.add_argument('--path_val_data2', type=str, default='', help='Specify the path to the validation data files directory')
parser.add_argument('--path_val_gts2', type=str, default='', help='Specify the path to the validation gts files directory')
parser.add_argument('--path_train_data3', type=str, default='', help='Specify the path to the training data files directory')
parser.add_argument('--path_train_gts3', type=str, default='', help='Specify the path to the training gts files directory')
parser.add_argument('--path_val_data3', type=str, default='', help='Specify the path to the validation data files directory')
parser.add_argument('--path_val_gts3', type=str, default='', help='Specify the path to the validation gts files directory')
parser.add_argument('--path_train_data4', type=str, default='', help='Specify the path to the training data files directory')
parser.add_argument('--path_train_gts4', type=str, default='', help='Specify the path to the training gts files directory')
parser.add_argument('--path_val_data4', type=str, default='', help='Specify the path to the validation data files directory')
parser.add_argument('--path_val_gts4', type=str, default='', help='Specify the path to the validation gts files directory')
parser.add_argument('--path_save', type=str, default='', help='Specify the path to the trained model will be saved')

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

    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    root_dir = args.path_save  # Path where the trained model is saved

    flair1 = sorted(glob(os.path.join(args.path_train_data1, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs1 = sorted(glob(os.path.join(args.path_train_gts1, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))                   # Collect all corresponding ground truths

    flair_val1 = sorted(glob(os.path.join(args.path_val_data1, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs_val1 = sorted(glob(os.path.join(args.path_val_gts1, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))                   # Collect all corresponding ground truths


    flair2 = sorted(glob(os.path.join(args.path_train_data2, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs2 = sorted(glob(os.path.join(args.path_train_gts2, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))                   # Collect all corresponding ground truths

    flair_val2 = sorted(glob(os.path.join(args.path_val_data2, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs_val2 = sorted(glob(os.path.join(args.path_val_gts2, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))                   # Collect all corresponding ground truths


    flair3 = sorted(glob(os.path.join(args.path_train_data3, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs3 = sorted(glob(os.path.join(args.path_train_gts3, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))                   # Collect all corresponding ground truths

    flair_val3 = sorted(glob(os.path.join(args.path_val_data3, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs_val3 = sorted(glob(os.path.join(args.path_val_gts3, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))                   # Collect all corresponding ground truths


    flair4 = sorted(glob(os.path.join(args.path_train_data4, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs4 = sorted(glob(os.path.join(args.path_train_gts4, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))                   # Collect all corresponding ground truths

    flair_val4 = sorted(glob(os.path.join(args.path_val_data4, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs_val4 = sorted(glob(os.path.join(args.path_val_gts4, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i)))                   # Collect all corresponding ground truths


    train_files=[]
    val_files=[]
    for i in range(len(flair1)):
        train_files = train_files + [{"image": fl,"label": seg} for fl, seg in zip(flair1[i:i+1], segs1[i:i+1])]
    for j in range(len(flair_val1)):
        val_files = val_files + [{"image": fl,"label": seg} for fl, seg in zip(flair_val1[j:j+1], segs_val1[j:j+1])]

    for i in range(len(flair2)):
        train_files = train_files + [{"image": fl,"label": seg} for fl, seg in zip(flair2[i:i+1], segs2[i:i+1])]
    for j in range(len(flair_val2)):
        val_files = val_files + [{"image": fl,"label": seg} for fl, seg in zip(flair_val2[j:j+1], segs_val2[j:j+1])]

    for i in range(len(flair3)):
        train_files = train_files + [{"image": fl,"label": seg} for fl, seg in zip(flair3[i:i+1], segs3[i:i+1])]
    for j in range(len(flair_val3)):
        val_files = val_files + [{"image": fl,"label": seg} for fl, seg in zip(flair_val3[j:j+1], segs_val3[j:j+1])]

    for i in range(len(flair4)):
        train_files = train_files + [{"image": fl,"label": seg} for fl, seg in zip(flair4[i:i+1], segs4[i:i+1])]
    for j in range(len(flair_val4)):
        val_files = val_files + [{"image": fl,"label": seg} for fl, seg in zip(flair_val4[j:j+1], segs_val4[j:j+1])]


    print("Training cases:", len(train_files))
    print("Validation cases:", len(val_files))
    
    # The below set of transformations perform the data augmentation

    train_transforms = Compose(
    [
        LoadNiftid(keys=["image","label"]),

        AddChanneld(keys=["image","label"]),
        Spacingd(keys=["image","label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        RandShiftIntensityd(keys="image",offsets=0.1,prob=1.0),
        RandScaleIntensityd(keys="image",factors=0.1,prob=1.0),
        RandCropByPosNegLabeld(keys=["image", "label"],label_key="label",spatial_size=(128, 128, 128),
            pos=4,neg=1,num_samples=32,image_key="image"),
        RandSpatialCropd(keys=["image", "label"], roi_size=(96,96,96), random_center=True, random_size=False),
        RandFlipd (keys=["image", "label"],prob=0.5,spatial_axis=(0,1,2)),
        RandRotate90d (keys=["image", "label"],prob=0.5,spatial_axes=(0,1)),
        RandRotate90d (keys=["image", "label"],prob=0.5,spatial_axes=(1,2)),
        RandRotate90d (keys=["image", "label"],prob=0.5,spatial_axes=(0,2)),
        RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
                     rotate_range=(np.pi/12, np.pi/12, np.pi/12), scale_range=(0.1, 0.1, 0.1),padding_mode='border'),
        ToTensord(keys=["image", "label"]),
    ]
    )
    val_transforms = Compose(
    [
        LoadNiftid(keys=["image", "label"]),
        
        #SqueezeDimd(keys=["flair", "mprage"], dim=-1),
        
        AddChanneld(keys=["image","label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "label"]),
    ]
    )

    #%%
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_train_ds = CacheDataset(data=train_files, transform=val_transforms, cache_rate=0.5, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    val_train_loader = DataLoader(val_train_ds, batch_size=1, num_workers=0)
  
    # device = torch.device("cuda:0")
    model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=0).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True, sigmoid=False,
                             include_background=False)

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    
    epoch_num = args.n_epochs
    val_interval = 5
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    metric_values_train = list()
    act = Activations(softmax=True)
    thresh = 0.4

    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:

            n_samples = batch_data["image"].size(0)
            for m in range(0,batch_data["image"].size(0),2):
                step += 2
                inputs, labels = (
                    batch_data["image"][m:(m+2)].to(device),
                    batch_data["label"][m:(m+2)].type(torch.LongTensor).to(device))
                optimizer.zero_grad()
                outputs = model(inputs)
                
                #Dice loss
                loss1 = loss_function(outputs,labels)
                
                #Focal loss
                ce_loss = nn.CrossEntropyLoss(reduction='none')
                ce = ce_loss((outputs),torch.squeeze(labels,dim=1))
                gamma = 2.0
                pt = torch.exp(-ce)
                f_loss = 1*(1-pt)**gamma * ce 
                loss2=f_loss
                loss2 = torch.mean(loss2)
                loss = 0.5*loss1+loss2              
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                if step%100 == 0:
                    step_print = int(step/2)
                    print(f"{step_print}/{(len(train_ds)*n_samples) // (train_loader.batch_size*2)}, train_loss: {loss.item():.4f}")

        epoch_loss /= step_print
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                                )
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model,mode='gaussian')
                   
                    val_labels = val_labels.cpu().numpy()
                    gt = np.squeeze(val_labels)
                    val_outputs = act(val_outputs).cpu().numpy()
                    seg= np.squeeze(val_outputs[0,1])
                    seg[seg>thresh]=1
                    seg[seg<thresh]=0
                    value = (np.sum(seg[gt==1])*2.0) / (np.sum(seg) + np.sum(gt))

                    metric_count += 1
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "Best_model_finetuning.pth"))
                    print("saved new best metric model")
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                                    f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                                    )
 
          
#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)