#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 08:59:13 2022

@author: Nataliia
"""

import torch

class GeneralizedLoss(torch.nn.Module):
    def __init__(self, loss_name, loss_one_class=None, loss_per_class=None):
        super().__init__()
        self.one_class = False
        self.mult_class = False
        if loss_name == 'focal_per_class':
            self.loss_function = focal_per_class
        elif loss_name == 'BCEWL_per_class':
            self.loss_function = BCEWL_per_class
        elif loss_name == 'BCEWL_one_class':
            self.loss_function = BCEWL_one_class
        elif loss_one_class is not None:
            self.loss_function = loss_one_class
            self.one_class = True
        elif loss_per_class is not None:
            self.loss_function = loss_per_class
            self.mult_class = True
        
    def forward(self, outputs, labels, device):
        if self.one_class:
            labels = (labels != 0.0).type(torch.LongTensor).to(device)
        elif self.mult_class:
            labels = torch.stack([
                (labels[:,0,:,:,:] == i).type(torch.LongTensor) for i in [1, 2]
                ], dim=1).to(device)
        
        return self.loss_function(outputs, labels, device)

def focal_per_class(outputs, labels, device):
    # 3 classes
    bg_loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs[:, 0:1, :, :, :],
                                           (labels == 0).type(torch.FloatTensor).to(device))
    cl_loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs[:, 2:3, :, :, :],
                                           (labels == 2).type(torch.FloatTensor).to(device))
    wm_loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs[:, 1:2, :, :, :],
                                           (labels == 1).type(torch.FloatTensor).to(device))
    
    gamma = torch.tensor(0.1, device=device)
    cl_loss = torch.pow(1-torch.exp(-cl_loss), gamma) * cl_loss
    wm_loss = torch.pow(1-torch.exp(-wm_loss), gamma) * wm_loss
    bg_loss = torch.pow(1-torch.exp(-bg_loss), gamma) * bg_loss
    
    cl_w = torch.tensor(5.0, device = device)
    loss_sum = cl_loss * cl_w + wm_loss + bg_loss

    return torch.mean(loss_sum, [1, 2, 3, 4]).mean()


def BCEWL_per_class(outputs, labels, device):
    # 3 classes
    bg_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(outputs[:, 0:1, :, :, :],
                                           (labels == 0).type(torch.FloatTensor).to(device))
    cl_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(outputs[:, 2:3, :, :, :],
                                           (labels == 2).type(torch.FloatTensor).to(device))
    wm_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(outputs[:, 1:2, :, :, :],
                                           (labels == 1).type(torch.FloatTensor).to(device))
    
    cl_w = torch.tensor(5.0, device = device)
    loss_sum = cl_loss * cl_w + wm_loss + bg_loss

    return torch.sum(loss_sum, [1, 2, 3, 4]).mean()

def BCEWL_one_class(outputs, labels, device):
    weight = (labels == 2).type(torch.FloatTensor) * 50 + (labels == 1).type(torch.FloatTensor) * 5 + 1
    weight = weight.to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(weight=weight)
    labels = (labels != 0).type(torch.FloatTensor).to(device)
    return loss_function(outputs, labels)
    