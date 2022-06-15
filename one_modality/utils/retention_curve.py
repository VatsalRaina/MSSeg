#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:46:03 2022

@author: meri
"""

import numpy as np
from joblib import Parallel, delayed
from functools import partial
from scipy import ndimage
from scipy.interpolate import interp1d
from collections import Counter
from .metrics import intersection_over_union

def cc_uncertainty(cc_mask, uncs_map, uncs_multi_mask, ens_pred):
    ''' from the cc mask'''
    les = uncs_map * cc_mask
    les = les[les != 0.0]
    res = {"sum": np.sum(uncs_map * cc_mask),
            "mean": np.sum(uncs_map * cc_mask) / np.sum(cc_mask),
            "logsum":np.log(np.sum(uncs_map * cc_mask)),
            "median": np.median(les),
            'volume': np.sum(cc_mask)}
    
    ''' from extended cc mask '''
    # find a connected component on the uncs_multi_mask with max intersection with cc_mask, aka lesion
    max_label = None
    max_iou = 0.0
    for intersec_label in np.unique(cc_mask * uncs_multi_mask):
        if intersec_label != 0.0:
            iou = intersection_over_union(cc_mask,
                                          (uncs_multi_mask == intersec_label).astype("float"))
            # print(iou)
            if iou > max_iou:
                max_iou = iou
                max_label = intersec_label
                # print(max_label, iou)
    if max_iou != 0.0:
        max_uncs_mask = (uncs_multi_mask == max_label).astype("float") # connected component with max intersection
            
        les = uncs_map * max_uncs_mask
        les = les[les != 0.0]
        res.update({"sum_ext": np.sum(uncs_map * max_uncs_mask),
                "mean_ext": np.sum(uncs_map * max_uncs_mask) / np.sum(max_uncs_mask),
                "logsum_ext":np.log(np.sum(uncs_map * max_uncs_mask)),
                "median_ext": np.median(les),
                'volume_ext': np.sum(max_uncs_mask)})
    else:
        res.update({"sum_ext": 0.0,
                "mean_ext": 0.0,
                "logsum_ext": 0.0,
                "median_ext": 0.0,
                'volume_ext': 0.0})
    
    ''' detection uncertainties '''
    # 1 - mean IoU of connected components on the ensemble's models prediction with cc_mask
    ens_ious = []
    for m in range(ens_pred.shape[0]):
        multi_mask = ndimage.label(ens_pred[m])[0]
        max_iou = 0.0
        for intersec_label in np.unique(cc_mask * multi_mask):
            if intersec_label != 0.0:
                iou = intersection_over_union(cc_mask, 
                                              (multi_mask == intersec_label).astype(float))
                if iou > max_iou:
                    max_iou = iou
        ens_ious.append(max_iou)
    res.update({'mean_iou_det': 1 - np.mean(ens_ious)})
    
    # 1 - intersection between the ensemble's models predictions and cc_mask aka lesion
    # don't compute mean IoU, but compute intersection and union between evaything
    intersec = cc_mask.copy()
    union = cc_mask.copy()
    for m in range(ens_pred.shape[0]):
        intersec *= ens_pred[m]
        union += ens_pred[m] - intersec
    res.update({'iou_ap_det': 1 - np.sum(intersec) / np.sum(union)})
    
    return res


def lesions_uncertainty(uncs_map, binary_mask, uncs_multi_mask, ens_pred, parallel):
    """
    Parallel evaluation of all uncertainties
    """
    def ld_to_dl(ld):
        keys = list(ld[0].keys())
        dl = dict(zip(keys, [[] for _ in keys]))
        for el in ld:
            for k in keys:
                v = el[k]
                dl[k].append(v)
        return dl
    
    multi_mask, n_labels = ndimage.label(binary_mask)
    cc_labels = np.arange(1, n_labels + 1)
    process = partial(cc_uncertainty,
                      uncs_map=uncs_map, 
                      uncs_multi_mask=uncs_multi_mask,
                      ens_pred=ens_pred)
    les_uncs_list = parallel(delayed(process)(
        cc_mask=(multi_mask == cc_label).astype("float")
        ) for cc_label in cc_labels
    )   # returns lists of dictionaries, but need dictionary of lists
    
    if les_uncs_list == []:
        metrics = {
        'sum': [], 'mean': [], 'logsum': [], 'median': [], 'volume': []
        }
        for k in list(metrics.keys()).copy():
            metrics[k + '_ext'] = []
        metrics['iou_ap_det'] = []
        metrics['mean_iou_det'] = []
        return metrics
    
    return ld_to_dl(les_uncs_list)


def get_lesion_rc_with_fn(gts, preds, uncs, multi_uncs_mask, IoU_threshold, fracs_retained, 
                  n_jobs, all_outputs):
    """
    :type gts: numpy.ndarray [H, W, D]
    :type preds: numpy.ndarray [H, W, D]
    :type uncs: numpy.ndarray [H, W, D]
    :type IoU_threshold: float
    :type fracs_retained: numpy.ndarray [n_fr,]
    :return:
    :rtype:
    """
    def compute_f1(lesion_types):
        counter = Counter(lesion_types)
        if counter['tp'] + 0.5 * (counter['fp'] + counter['fn']) == 0.0:
            return 0
        return counter['tp'] / (counter['tp'] + 0.5 * (counter['fp'] + counter['fn']))
    
    def eval_rc(tp_uncs_l, fp_uncs_l, fn_uncs_l):
        # form a list of lesion uncertainties and lesion types
        uncs_list = [np.asarray(tp_uncs_l), 
                     np.asarray(fp_uncs_l), 
                     np.asarray(fn_uncs_l)]  # tp, fp, fn
    
        lesion_type_list = [np.full(shape=unc.shape, fill_value=fill)
                            for unc, fill in zip(uncs_list, ['tp', 'fp', 'fn'])
                            if unc.size > 0]
        uncs_list = [unc for unc in uncs_list if unc.size > 0]
        uncs_all = np.concatenate(uncs_list, axis=0)
        lesion_type_all = np.concatenate(lesion_type_list, axis=0)
    
        del uncs_list, lesion_type_list
        assert uncs_all.shape == lesion_type_all.shape
    
        # sort uncertainties and lesions types
        ordering = uncs_all.argsort()
        lesion_type_all = lesion_type_all[ordering][::-1]
    
        f1_values = [compute_f1(lesion_type_all)]
    
        # reject the most uncertain lesion
        for i_l, lesion_type in enumerate(lesion_type_all):
            if lesion_type == 'fp':
                lesion_type_all[i_l] = 'tn'
            elif lesion_type == 'fn':
                lesion_type_all[i_l] = 'tp'
            f1_values.append(compute_f1(lesion_type_all))
    
        # interpolate the curve and make predictions in the retention fraction nodes
        n_lesions = lesion_type_all.shape[0]
        spline_interpolator = interp1d(x=[_ / n_lesions for _ in range(n_lesions + 1)], 
                                       y=f1_values[::-1],
                                       kind='slinear', fill_value="extrapolate")
        return spline_interpolator(fracs_retained), f1_values[::-1]

    from .transforms import get_TP_FP_lesions_mask_parallel, get_FN_lesions_mask_parallel
    
    les_uncs_rc = {
        'sum': None, 'mean': None, 'logsum': None, 'median': None, 'volume': None
    }
    for k in list(les_uncs_rc.keys()).copy():
        les_uncs_rc[k + '_ext'] = None
    les_uncs_rc['iou_ap_det'] = None
    les_uncs_rc['mean_iou_det'] = None
    f1_nodes = les_uncs_rc.copy()
    
    with Parallel(n_jobs=n_jobs) as parallel_backend:
        fn_lesions = get_FN_lesions_mask_parallel(ground_truth=gts,
                                            predictions=preds,
                                            IoU_threshold=IoU_threshold,
                                            mask_type='binary',
                                            parallel_backend=parallel_backend)
        
        tp_lesions, fp_lesions = get_TP_FP_lesions_mask_parallel(ground_truth=gts,
                                                  predictions=preds,
                                                  IoU_threshold=IoU_threshold,
                                                  mask_type='binary',
                                                  parallel_backend=parallel_backend)
        
        tp_uncs = lesions_uncertainty(uncs_map=uncs,
                                    binary_mask=tp_lesions,
                                    uncs_multi_mask=multi_uncs_mask, 
                                    ens_pred=all_outputs,
                                    parallel=parallel_backend)
    
        fp_uncs = lesions_uncertainty(uncs_map=uncs,
                                    binary_mask=fp_lesions,
                                    uncs_multi_mask=multi_uncs_mask, 
                                    ens_pred=all_outputs,
                                    parallel=parallel_backend)
        
        fn_uncs = lesions_uncertainty(uncs_map=uncs,
                                    binary_mask=fn_lesions,
                                    uncs_multi_mask=multi_uncs_mask, 
                                    ens_pred=all_outputs,
                                    parallel=parallel_backend)
        
        del tp_lesions, fp_lesions, fn_lesions
    
        res = parallel_backend(delayed(eval_rc)(tp_uncs_l=tp_uncs[ut], 
                                                fp_uncs_l=fp_uncs[ut], 
                                                fn_uncs_l=fn_uncs[ut])
                               for ut in les_uncs_rc.keys())
    
    for ut, res_ut in zip(list(les_uncs_rc.keys()), res):
        les_uncs_rc[ut], f1_nodes[ut] = res_ut
    
    return les_uncs_rc, f1_nodes