"""
Implementation of standard predictive uncertainty measures for image segmentation
"""

import numpy as np
from scipy import ndimage

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


def lesions_uncertainty_sum(uncs_mask, binary_mask):
    """
    Binary mask can be either gt or pred.

    Returns:
        * list of uncertainties computed as sum of connected component uncertainty
        in voxels normalized by the number of voxels
        * connected components map from binary mask

    index in cc map = index in uncertainty list

    index 0 in uncertainty list is background
    """
    multi_mask = ndimage.label(binary_mask)        # connected components map
    uncs_list = []
    for cc_label in np.unique(multi_mask):
        cc_mask = (multi_mask == cc_label).astype("float32")
        cc_unc = np.sum(uncs_mask * cc_mask) / np.sum(cc_mask)
        uncs_list.append(cc_unc)
    return uncs_list, multi_mask
