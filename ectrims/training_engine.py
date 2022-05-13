#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:28:52 2022

@author: meri
"""
import torch
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn

def compute_meandice(y_pred, y):
    device = y_pred.device
    if y_pred.size() != y.size():
        raise ValueError(f"prediction {y_pred.size()}, target {y.size()}")
    if y_pred.dim() != 5 or y.dim() != 5:
        raise ValueError(f"Exprect five dims, prediction {y_pred.size()}, target {y.size()}")
    # if np.isin(y_pred.unique().cpu().numpy(), [0, 1]):
    #     raise ValueError(f"y_pred should be OHE, got values {y_pred.unique()}")
    # if np.isin(y.unique().cpu().numpy(), [0, 1]):
    #     raise ValueError(f"y should be OHE, got values {y.unique()}")
        
    b = y.size(0)
    
    mean_dice = torch.tensor(0.0, device=device)
    for i_b in range(b):
        denominator = y_pred[i_b].sum() + y[i_b].sum()
        intersec = torch.sum(y_pred[i_b] * y[i_b])
        if denominator == torch.tensor(0.0, device=device):
            mean_dice += torch.tensor(1.0, device=device)
        else:
            mean_dice += 2.0 * intersec / denominator
    return mean_dice / b

def print_tensor_summary(t, name):
    print('Name: ', name)
    print('Size: ', t.size())
    print('Unique: ', t.unique())
    print('Sum: ', t.sum())
    
def init_weights(unet):
    for layer in unet.model.modules():
        if type(layer) == nn.Conv3d:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
    return unet
            
def validation(model, act, val_loader, loss_function, device, thresh, only_loss=False):
    model.eval()
    with torch.no_grad():
        if not only_loss:
            metric_sum = 0.0
        metric_count = 0
        loss_sum = 0.0
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, mode='gaussian')

            val_labels = torch.stack([
            (val_labels[:,0,:,:,:] == i).type(torch.LongTensor) for i in [1, 2]
            ], dim=1).to(device)
            
            loss_sum += loss_function(val_outputs, val_labels).item()
            
            val_outputs = act(val_outputs)
            val_outputs[val_outputs >= thresh] = 1.
            val_outputs[val_outputs < thresh] = 0.

            metric_count += 1
            if not only_loss:
                metric_sum += compute_meandice(val_outputs, val_labels).mean().item()
    model.train(True)
    if only_loss:          
        return loss_sum / metric_count
    return loss_sum / metric_count, metric_sum / metric_count


def validation_one_class(model, act, val_loader, loss_function, device, thresh, only_loss=False):
    model.eval()
    with torch.no_grad():
        if not only_loss:
            metric_sum = 0.0
        metric_count = 0
        loss_sum = 0.0
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, mode='gaussian')

            val_labels = (val_labels > 0).type(torch.LongTensor).to(device)
            
            loss_sum += loss_function(val_outputs, val_labels).item()
            
            val_outputs = act(val_outputs)
            val_outputs[val_outputs >= thresh] = 1.
            val_outputs[val_outputs < thresh] = 0.
              

            metric_count += 1
            if not only_loss:
                metric_sum += compute_meandice(val_outputs, val_labels).mean().item()
    model.train(True)
    if only_loss:          
        return loss_sum / metric_count
    return loss_sum / metric_count, metric_sum / metric_count


def plot_history(epoch_loss_values, val_loss_values, lrs, metric_values, metric_values_train, val_interval, save_path):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].set_title("Epoch Average Train Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    x1 = [i + 1 for i, _ in enumerate(val_loss_values)]
    y1 = val_loss_values
    ax[0].set_xlabel("epoch")
    ax[0].plot(x, y, label='train')
    ax[0].plot(x1, y1, label='val')
    ax[0].legend()
    
    ax[1].set_title("Learning rate")
    ax[1].plot([i + 1 for i, _ in enumerate(lrs)], lrs)
    ax[1].set_xlabel("epoch")
    
    ax[2].set_title("Val and Train Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    y1 = metric_values_train
    ax[2].set_xlabel("epoch")
    ax[2].plot(x, y, label='val')
    ax[2].plot(x, y1, label='train')
    ax[2].legend()
    
    fig.savefig(os.path.join(save_path, 'train_history.png'))
    plt.close(fig)
    
'''    
def train_one_epoch(model, train_loader, device, optimizer, scheduler, loss_function, epoch, act, val_loader, thresh):
    model.train()
    epoch_loss = 0
    epoch_loss_val = []
    step = 0
    for batch_data in train_loader:
        n_samples = batch_data["image"].size(0)
        for m in range(0, batch_data["image"].size(0), 2):
            step += 2
            inputs, labels = (
                batch_data["image"][m:(m + 2)].to(device),
                batch_data["label"][m:(m + 2)])
            optimizer.zero_grad()
            outputs = model(inputs)

            # Dice loss
            labels = torch.stack([
                (labels[:,0,:,:,:] == i).type(torch.LongTensor) for i in [0, 1, 2]
                ], dim=1).to(device)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 100 == 0:
                step_print = int(step / 2)
                val_loss = validation(model, act, val_loader, loss_function, device, thresh, only_loss=True)
                epoch_loss_val.append(val_loss)
                print(
                    f"{step_print}/{(len(train_loader) * n_samples) // (train_loader.batch_size * 2)}, train_loss: {loss.item():.4f}")
    lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    epoch_loss /= step_print
    
    return model, lr , epoch_loss, np.mean(epoch_loss_val), optimizer, scheduler
    '''

def train_one_epoch(model, train_loader, device, optimizer, scheduler, loss_function, epoch, act, val_loader, thresh):
    model.train(True)
    epoch_loss = 0
    step = 0
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    for batch_data in train_loader:
        n_samples = batch_data["image"].size(0)
        for m in range(0, batch_data["image"].size(0), 2):
            step += 2
            inputs, labels = (
                batch_data["image"][m:(m + 2)].to(device),
                batch_data["label"][m:(m + 2)])
            optimizer.zero_grad()
            outputs = model(inputs)

            # Dice loss
            labels = torch.stack([
                (labels[:,0,:,:,:] == i).type(torch.LongTensor) for i in [1, 2]
                ], dim=1).to(device)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 100 == 0:
                step_print = int(step / 2)
                val_loss, val_metric = validation(model, act, val_loader, loss_function, device, thresh, only_loss=False)
                print(
                    f"{step_print}/{(len(train_loader) * n_samples) // (train_loader.batch_size * 2)}, train_loss: {loss.item():.4f}"
                    f"\nval_loss {val_loss:.4f}"
                    )
    lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    epoch_loss /= step_print
    
    return lr, epoch_loss


def train_one_epoch_one_class(model, train_loader, device, optimizer, scheduler, loss_function, epoch, act, val_loader, thresh):
    model.train(True)
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        n_samples = batch_data["image"].size(0)
        for m in range(0, batch_data["image"].size(0), 2):
            step += 2
            inputs, labels = (
                batch_data["image"][m:(m + 2)].to(device),
                batch_data["label"][m:(m + 2)])
            optimizer.zero_grad()
            outputs = model(inputs)

            # Dice loss
            labels = (labels > 0).type(torch.LongTensor).to(device)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 100 == 0:
                step_print = int(step / 2)
                val_loss, val_metric = validation_one_class(model, act, val_loader, loss_function, device, thresh, only_loss=False)
                print(
                    f"{step_print}/{(len(train_loader) * n_samples) // (train_loader.batch_size * 2)}, train_loss: {loss.item():.4f}"
                    f"\nval_loss {val_loss:.4f}"
                    )
    lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    epoch_loss /= step_print
    
    return lr, epoch_loss
    
    
    
    