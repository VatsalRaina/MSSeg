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
from monai.transforms import Activations
import numpy as np
import random
from data_loader import train_transforms, val_transforms, get_data_loader

'''
python Training.py \
--learning_rate LEARNING_RATE \
--n_epochs N_EPOCHS \
--seed SEED \
--threshold THRESHOLD\
--path_flair PATH_FLAIR \
--path_mp2rage PATH_MP2RAGE \
--path_gts PATH_GTS \
[--flair_prefix FLAIR_PREFIX \
--mp2rage_prefix MP2RAGE_PREFIX\ 
--gts_prefix GTS_PREFIX \
--check_dataset \
--num_workers NUM_WORKERS \
--path_save PATH_SAVE

python Training.py \
--learning_rate 1e-4 \
--n_epochs 1 \
--seed 1 \
--threshold 0.4 \
--path_flair /Users/nataliyamolchanova/Docs/phd_application/MSxplain/data/canonical/dev_in \
--path_mp2rage /Users/nataliyamolchanova/Docs/phd_application/MSxplain/data/canonical/dev_in \
--path_gts /Users/nataliyamolchanova/Docs/phd_application/MSxplain/data/canonical/dev_in \
--flair_prefix FLAIR.gt.nii \
--mp2rage_prefix FLAIR.gt.nii \
--gts_prefix gt.nii \
--check_dataset \
--num_workers 0 \
--path_save /Users/nataliyamolchanova/Docs/phd_application/MSxplain/results/tmp
'''

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# training
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify the initial learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='Specify the number of epochs to train for')
# model
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--threshold', type=float, default=0.4, help='Threshold for lesion detection')
# data
parser.add_argument('--path_train', type=str, required=True, help='Specify the path to the train data directory')
parser.add_argument('--path_val', type=str, default='', help='Specify the path to the val data directory')
parser.add_argument('--flair_prefix', type=str, default="FLAIR.nii.gz", help='name ending FLAIR')
parser.add_argument('--mp2rage_prefix', type=str, default="UNIT1.nii.gz", help='name ending mp2rage')
parser.add_argument('--gts_prefix', type=str, default="gt.nii", help='name ending segmentation mask')
parser.add_argument('--check_dataset', action='store_true',
                    help='if sent, checks that FLAIR and semg masks names correspond to each other')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of jobs used to load the data, DataLoader parameter')
# save
parser.add_argument('--path_save', type=str, default='', help='Specify the path to the save directory')

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):
    os.makedirs(args.path_save, exist_ok=True)

    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' Choose device'''
    device = get_default_device()
    val_transforms_seed = val_transforms.set_random_state(seed=seed_val)
    train_transforms_seed = train_transforms.set_random_state(seed=seed_val)

    ''' Dataloader '''
    train_loader = get_data_loader(path_flair=args.path_train,
                                   path_mp2rage=args.path_train,
                                   path_gts=args.path_train,
                                   flair_prefix=args.flair_prefix,
                                   mp2rage_prefix=args.mp2rage_prefix,
                                   gts_prefix=args.gts_prefix,
                                   transforms=train_transforms_seed,
                                   num_workers=args.num_workers,
                                   batch_size=1)
    val_loader = get_data_loader(path_flair=args.path_val,
                                 path_mp2rage=args.path_val,
                                 path_gts=args.path_val,
                                 flair_prefix=args.flair_prefix,
                                 mp2rage_prefix=args.mp2rage_prefix,
                                 gts_prefix=args.gts_prefix,
                                 transforms=val_transforms_seed,
                                 num_workers=args.num_workers,
                                 batch_size=1)
    val_train_loader = get_data_loader(path_flair=args.path_train,
                                   path_mp2rage=args.path_train,
                                   path_gts=args.path_train,
                                   flair_prefix=args.flair_prefix,
                                   mp2rage_prefix=args.mp2rage_prefix,
                                   gts_prefix=args.gts_prefix,
                                   transforms=val_transforms_seed,
                                   num_workers=args.num_workers,
                                   batch_size=1)

    ''' Init model '''
    model = UNet(
        dimensions=3,
        in_channels=2,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=0).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True,
                             sigmoid=False,
                             include_background=False)

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    # Load trained model
    # model.load_state_dict(torch.load(os.path.join(root_dir, "Initial_model.pth")))

    epoch_num = args.n_epochs
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    metric_values_train = list()
    act = Activations(softmax=True)
    thresh = args.threshold

    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            n_samples = batch_data["image"].size(0)
            for m in range(0, batch_data["image"].size(0), 2):
                step += 2
                inputs, labels = (
                    batch_data["image"][m:(m + 2)].to(device),
                    batch_data["label"][m:(m + 2)].type(torch.LongTensor).to(device))
                optimizer.zero_grad()
                outputs = model(inputs)

                # Dice loss
                loss1 = loss_function(outputs, labels)

                # Focal loss
                ce_loss = nn.CrossEntropyLoss(reduction='none')
                ce = ce_loss(outputs, torch.squeeze(labels, dim=1))
                gamma = 2.0
                pt = torch.exp(-ce)
                f_loss = 1 * (1 - pt) ** gamma * ce
                loss2 = f_loss
                loss2 = torch.mean(loss2)
                loss = 0.5 * loss1 + loss2

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if step % 200 == 0:
                    step_print = int(step / 2)
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
                            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, mode='gaussian')
                            
                            loss1 = loss_function(val_outputs, val_labels)

                            # Focal loss
                            ce_loss = nn.CrossEntropyLoss(reduction='none')
                            ce = ce_loss(outputs, torch.squeeze(labels, dim=1))
                            gamma = 2.0
                            pt = torch.exp(-ce)
                            f_loss = 1 * (1 - pt) ** gamma * ce
                            loss2 = f_loss
                            loss2 = torch.mean(loss2)
                            loss = 0.5 * loss1 + loss2
        
                            val_labels = val_labels.cpu().numpy()
                            gt = np.squeeze(val_labels)
                            val_outputs = act(val_outputs).cpu().numpy()
                            seg = np.squeeze(val_outputs[0, 1])
                            seg[seg > thresh] = 1
                            seg[seg < thresh] = 0
                            value = (np.sum(seg[gt == 1]) * 2.0) / (np.sum(seg) + np.sum(gt))
        
                            metric_count += 1
                            metric_sum += value.sum().item()
                        val_metric = metric_sum / metric_count
                        val_loss = loss.item()
                    model.train(True)
                    print(
                        f"{step_print}/{(len(train_loader) * n_samples) // (train_loader.batch_size * 2)}, train_loss: {loss.item():.4f}, "
                        f"val_loss: {val_loss:.4f}, val_metric: {val_metric:.4f}"
                        )

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
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, mode='gaussian')

                    val_labels = val_labels.cpu().numpy()
                    gt = np.squeeze(val_labels)
                    val_outputs = act(val_outputs).cpu().numpy()
                    seg = np.squeeze(val_outputs[0, 1])
                    seg[seg > thresh] = 1
                    seg[seg < thresh] = 0
                    value = (np.sum(seg[gt == 1]) * 2.0) / (np.sum(seg) + np.sum(gt))

                    metric_count += 1
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                metric_sum_train = 0.0
                metric_count_train = 0
                for train_data in val_train_loader:
                    train_inputs, train_labels = (
                        train_data["image"].to(device),
                        train_data["label"].to(device),
                    )
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    train_outputs = sliding_window_inference(train_inputs, roi_size, sw_batch_size, model,
                                                              mode='gaussian')
                
                    train_labels = train_labels.cpu().numpy()
                    gt = np.squeeze(train_labels)
                    train_outputs = act(train_outputs).cpu().numpy()
                    seg = np.squeeze(train_outputs[0, 1])
                    seg[seg > thresh] = 1
                    seg[seg < thresh] = 0
                    value_train = (np.sum(seg[gt == 1]) * 2.0) / (np.sum(seg) + np.sum(gt))
                
                    metric_count_train += 1
                    metric_sum_train += value_train.sum().item()
                metric_train = metric_sum_train / metric_count_train
                metric_values_train.append(metric_train)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(args.path_save, "Best_model_finetuning.pth"))
                    print("saved new best metric model")
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                      f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                      )
                plt.figure("train", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Epoch Average Train Loss")
                x = [i + 1 for i in range(len(epoch_loss_values))]
                y = epoch_loss_values
                plt.xlabel("epoch")
                plt.plot(x, y)
                plt.subplot(1, 2, 2)
                plt.title("Val and Train Mean Dice")
                x = [val_interval * (i + 1) for i in range(len(metric_values))]
                y = metric_values
                y1 = metric_values_train
                plt.xlabel("epoch")
                plt.plot(x, y)
                plt.plot(x, y1)
                plt.savefig(os.path.join(args.path_save, 'train_history.png'))
                plt.show()

            # %%


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)