#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:56:15 2022

@author: meri
"""

"""
Implementation of standard predictive uncertainty measures for image segmentation
"""

import numpy as np
from scipy import ndimage
from scipy.stats import multivariate_normal
# 


def entropy_of_expected(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_voxels_X, num_voxels_Y, num_voxels_Z, num_classes]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=-1)

def expected_entropy(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_voxels_X, num_voxels_Y, num_voxels_Z, num_classes]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    log_probs = -np.log(probs + epsilon)
    return np.mean(np.sum(probs * log_probs, axis=-1), axis=0)


def ensemble_uncertainties_classification(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_voxels_X, num_voxels_Y, num_voxels_Z, num_classes]
    :return: Dictionary of uncertainties
    """
    mean_probs = np.mean(probs, axis=0)
    mean_lprobs = np.mean(np.log(probs + epsilon), axis=0)
    conf = np.max(mean_probs, axis=-1)

    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)

    mutual_info = eoe - exe

    epkl = -np.sum(mean_probs * mean_lprobs, axis=-1) - exe

    uncertainty = {'confidence': -1 * conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'epkl': epkl,
                   'reverse_mutual_information': epkl - mutual_info
                   }

    return uncertainty


def lesions_uncertainty_sum(uncs_mask, binary_mask, dtype="float32", mask_type='one_hot', unc_type='average'):
    """
    Binary mask can be either gt or pred.

    Returns:
        * list of uncertainties computed as sum of connected component uncertainty
        in voxels normalized by the number of voxels
        * connected components map from binary mask

    index in cc map = index in uncertainty list

    index 0 in uncertainty list is background
    """
    from .transforms import mask_encoding
    
    multi_mask = ndimage.label(binary_mask)[0]        # connected components map
    uncs_list = []
    lesions = []
    for cc_label in np.unique(multi_mask):
        if cc_label != 0.0:                         # exclude background
            cc_mask = (multi_mask == cc_label).astype(dtype)
            #if np.sum(cc_mask) > n_min_vox:
            if unc_type=="average":
                cc_unc = np.sum(uncs_mask * cc_mask) / np.sum(cc_mask)
            elif unc_type=="sum":
                cc_unc = np.sum(uncs_mask * cc_mask)
            elif unc_type=="count":
                cc_unc = np.sum(cc_mask)
            elif unc_type=="central":
                # identify the central voxel and only use that as the uncertainty measure
                voxel_pos = np.where(cc_mask==1)
                posX = int(np.median(voxel_pos[0]))
                posY = int(np.median(voxel_pos[1]))
                posZ = int(np.median(voxel_pos[2]))
                cc_unc = uncs_mask[posX,posY,posZ]
            elif unc_type=="gaussian":
                # weight the uncertainties using a 3D Gaussian centred at the lesion centre and then just add them up
                voxel_pos = np.where(cc_mask==1)
                posX = int(np.median(voxel_pos[0]))
                posY = int(np.median(voxel_pos[1]))
                posZ = int(np.median(voxel_pos[2]))
                maxX, maxY, maxZ = np.shape(cc_mask)
                x, y, z = np.mgrid[0:maxX:1, 0:maxY:1, 0:maxZ:1]
                grid = np.stack((x,y,z), axis=-1)
                mean_vector = [posX, posY, posZ]
                covariance_matrix = [[1,0,0], [0,1,0], [0,0,1]]
                rv = multivariate_normal(mean_vector, covariance_matrix)
                gauss_mat = rv.pdf(grid)
                weighted_unc = gauss_mat * uncs_mask
                cc_unc = np.sum(weighted_unc * cc_mask)

            uncs_list.append(cc_unc)
            lesions.append(cc_mask)

    return np.asarray(uncs_list), mask_encoding(lesion_masks_list=lesions, dtype=dtype, mask_type=mask_type)


