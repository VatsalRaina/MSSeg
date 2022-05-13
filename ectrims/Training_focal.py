"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
"""

import argparse
import os

import torch

from monai.losses import DiceFocalLoss, FocalLoss
from monai.networks.nets import UNet
from monai.transforms import Activations
import numpy as np
import random
from data_loader import train_transforms, val_transforms, get_data_loader
from training_engine import *

'''
python Training.py \
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
                                   batch_size=1, cash_rate=0.5)
    val_loader = get_data_loader(path_flair=args.path_val,
                                 path_mp2rage=args.path_val,
                                 path_gts=args.path_val,
                                 flair_prefix=args.flair_prefix,
                                 mp2rage_prefix=args.mp2rage_prefix,
                                 gts_prefix=args.gts_prefix,
                                 transforms=val_transforms_seed,
                                 num_workers=args.num_workers,
                                 batch_size=1, cash_rate=0.5)
    val_train_loader = get_data_loader(path_flair=args.path_train,
                                   path_mp2rage=args.path_train,
                                   path_gts=args.path_train,
                                   flair_prefix=args.flair_prefix,
                                   mp2rage_prefix=args.mp2rage_prefix,
                                   gts_prefix=args.gts_prefix,
                                   transforms=val_transforms_seed,
                                   num_workers=args.num_workers,
                                   batch_size=1, cash_rate=0.5)

    ''' Init model '''
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=0)
    model = init_weights(model)
    model = model.to(device)
    # loss_function = DiceCELoss(ce_weight=torch.Tensor([1, 5]), lambda_dice=0.5, lambda_ce=1.)
    # loss_function = DiceFocalLoss(include_background=True,
    #                               to_onehot_y=False,
    #                               softmax=True,
    #                               focal_weight=torch.Tensor([1, 1, 5]), 
    #                               lambda_dice=0.5, 
    #                               lambda_focal=1.0, 
    #                               gamma=2.0,
    #                               reduction='mean')
    loss_function = FocalLoss(include_background=True, to_onehot_y=False, 
                              gamma=2.0, weight=torch.Tensor([1., 5.]), 
                              reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-5, 
                                                  max_lr=5e-3,step_size_up=10,
                                                  cycle_momentum=False,
                                                  mode="triangular2")
    # Load trained model
    # model.load_state_dict(torch.load(os.path.join(root_dir, "Initial_model.pth")))

    epoch_num = args.n_epochs
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    val_loss_values = list()
    metric_values = list()
    metric_values_train = list()
    lrs = list()
    act = Activations(softmax=True)
    thresh = args.threshold
    save_path=args.path_save
    
    # early stopping params
    patience = 5    # stop training if val loss did not improve during several epochs
    tolerance = 1e-6
    last_loss = np.inf
    current_loss = 0.0
    silence_epochs = 0.0

    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        lr , epoch_loss=train_one_epoch(model, train_loader, device, optimizer, scheduler, loss_function, epoch, act, val_loader, thresh)
        lrs.append(lr)
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average train loss: {epoch_loss:.4f}")
        
        # early stopping
        current_loss = validation(model, act, val_loader, loss_function, device, thresh, only_loss=True)
        if current_loss > last_loss or abs(current_loss - last_loss) < tolerance:
            silence_epochs += 1
            if silence_epochs > patience:
                print(f"Early stopping on epoch {epoch + 1}")
                break
        else:
            last_loss = current_loss
            
        # validation
        if (epoch + 1) % val_interval == 0:
            val_loss, val_dice = validation(model, act, val_loader, loss_function, device, thresh, only_loss=False)
            metric_values.append(val_dice)
            val_loss_values.append(val_loss)
            
            train_loss, train_dice = validation(model, act, val_train_loader, loss_function, device, thresh, only_loss=False)
            metric_values_train.append(train_dice)
            
            if val_dice > best_metric:
                best_metric = val_dice
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.path_save, "Best_model_finetuning.pth"))
                print("saved new best metric model")
            print(f"current epoch: {epoch + 1} current mean dice: {val_dice:.4f}"
                  f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                  )
            
        plot_history(epoch_loss_values, val_loss_values, lrs, metric_values, 
                     metric_values_train, val_interval, save_path) 

            # %%


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
