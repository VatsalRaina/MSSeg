# -*- coding: utf-8 -*-

"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
         Adapted by Nataliia Molchanova
         
"""

import argparse
import os
from functools import partial
import numpy as np
from pathlib import Path
import re
import pandas as pd
from scipy import ndimage
from joblib import Parallel, delayed
from utils.metrics import lesion_metrics_parallel, dice_norm_metric, intersection_over_union

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.35, help='Threshold for lesion detection')

parser.add_argument('--les_unc_measure', type=str, default="reverse_mutual_information",
                    help='name of a single uncertainty metric')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the test data files directory')
parser.add_argument('--path_save', type=str, default='', help='Specify the path to save the segmentations')
parser.add_argument('--n_jobs', type=int, default=1,
                    help="number of cores used for computation of dsc for different retention fractions")
parser.add_argument('--IoU_threshold', default=0.5, type=float, help="IoU threshold for lesion F1 computation")
parser.add_argument('--les_uncs_threshold', required=True, type=float, help="IoU threshold for lesion F1 computation")
parser.add_argument('--eval_initial', action='store_true')

def evaluate(gt, seg, IoU_threshold, parallel_backend):
    row = dict()
    # voxels scale metrics: nDSC, TPR, FPR, FNR
    row['nDSC'] = dice_norm_metric(ground_truth=gt.flatten(), 
                                    predictions=seg.flatten())[0]
    tp = np.sum(seg[gt == 1])
    fp = np.sum(seg[gt == 0])
    row['TPR'] = tp / np.sum(gt)
    row['FPR'] = fp / np.sum(gt == 0)
    row['FNR'] = 1. - row['TPR']
    # lesion scale metrics: F1, TPR, FPR, FNR
    f1, tpr, fdr, fnr = lesion_metrics_parallel(ground_truth=gt, 
                                                predictions=seg, 
                                                IoU_threshold=IoU_threshold, 
                                                parallel_backend=parallel_backend)
    row['F1 les'] = f1
    row['TPR les'] = tpr
    row['FDR les'] = fdr
    row['FNR les'] = fnr
    return row

def cc_uncertainty(cc_mask, uncs_type, 
                   uncs_map=None, 
                   uncs_multi_mask=None, 
                   ens_pred=None):
    if uncs_type == 'mean_iou_det':
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
        return 1 - np.mean(ens_ious)
    
    elif uncs_type == 'iou_ap_det':
        intersec = cc_mask.copy()
        union = cc_mask.copy()
        for m in range(ens_pred.shape[0]):
            intersec *= ens_pred[m]
            union += ens_pred[m] - intersec
        return 1 - np.sum(intersec) / np.sum(union)
    
    elif uncs_type[-len('_ext'):] == '_ext':
        max_label = None
        max_iou = 0.0
        for intersec_label in np.unique(cc_mask * uncs_multi_mask):
            if intersec_label != 0.0:
                iou = intersection_over_union(cc_mask,
                                              (uncs_multi_mask == intersec_label).astype("float"))
                if iou > max_iou:
                    max_iou = iou
                    max_label = intersec_label
        if max_iou != 0.0:
            max_uncs_mask = (uncs_multi_mask == max_label).astype("float") # connected component with max intersection
            les = uncs_map * max_uncs_mask
            les = les[les != 0.0]
            if uncs_type == "sum_ext": return np.sum(uncs_map * max_uncs_mask)
            elif uncs_type == "mean_ext": return np.sum(uncs_map * max_uncs_mask) / np.sum(max_uncs_mask)
            elif uncs_type == "logsum_ext": np.log(np.sum(uncs_map * max_uncs_mask))
            elif uncs_type == "median_ext": return np.median(les)
            elif uncs_type == 'volume_ext': return np.sum(max_uncs_mask)
            else: raise NotImplementedError(uncs_type)
        else:
            return 0.0
    else:
        les = uncs_map * cc_mask
        les = les[les != 0.0]
        if uncs_type == "sum": return np.sum(uncs_map * cc_mask)
        elif uncs_type == "mean": return np.sum(uncs_map * cc_mask) / np.sum(cc_mask)
        elif uncs_type == "logsum": return np.log(np.sum(uncs_map * cc_mask))
        elif uncs_type == "median": return np.median(les)
        elif uncs_type == 'volume': return np.sum(cc_mask)
        else: raise NotImplementedError(uncs_type)

def filter_lesions(uncs_map, binary_mask, uncs_type, uncs_threshold, parallel,
                        uncs_multi_mask=None, ens_pred=None):
    multi_mask, n_labels = ndimage.label(binary_mask)
    cc_labels = np.arange(1, n_labels + 1)
    process = partial(cc_uncertainty,
                      uncs_map=uncs_map, 
                      uncs_multi_mask=uncs_multi_mask,
                      ens_pred=ens_pred,
                      uncs_type=uncs_type)
    les_uncs_list = parallel(delayed(process)(
        cc_mask=(multi_mask == cc_label).astype("float")
        ) for cc_label in cc_labels
    )   # returns lists of dictionaries, but need dictionary of lists
    
    new_seg = np.zeros_like(binary_mask)
    for i_cc, cc_label in enumerate(cc_labels):
        if les_uncs_list[i_cc] < uncs_threshold:
            new_seg += (multi_mask == cc_label).astype("float")
    
    return new_seg
    
def main(args):
    npy_files = Path(args.path_data).glob("*data.npz")
        
    os.makedirs(args.path_save, exist_ok=True)
    
    # %%
    print("Staring evaluation")
    fracs_ret = np.log(np.arange(200 + 1)[1:])
    fracs_ret /= np.amax(fracs_ret)
    metrics = ['nDSC', 'TPR', 'FPR', 'FNR',
               'F1 les', 'TPR les', 'FDR les', 'FNR les',
               'hash']
    metrics_i = pd.DataFrame([], columns=metrics)
    metrics_f = metrics_i.copy()
    
    with Parallel(n_jobs=args.n_jobs) as parallel:
        for npy_file in npy_files:
            # loading variables
            npy_loader = np.load(npy_file)
            
            filename_or_obj = re.sub("_data.npz", "", os.path.basename(npy_file))
            
            # load the data
            gt=npy_loader['gt']
            seg=npy_loader['seg']
                    
            all_outputs = npy_loader['all_outputs']
            ens_seg = all_outputs.copy()
            ens_seg[ens_seg >= args.threshold] = 1.0
            ens_seg[ens_seg < args.threshold] = 0.0
            
            del all_outputs
            
            # evaluation full
            if args.eval_initial:
                metrics_row = evaluate(gt, seg, args.IoU_threshold, parallel)
                metrics_row['hash'] = filename_or_obj
                metrics_i = metrics_i.append(metrics_row, ignore_index=True)
                metrics_i.to_csv(os.path.join(args.path_save, "intial_metrics.csv"))
    
            # filtering based lesions uncertainty
            seg_filt = filter_lesions(uncs_map=npy_loader['uncs_value'], 
                                      binary_mask=seg, 
                                      uncs_type=args.les_unc_measure, 
                                      uncs_threshold=args.les_uncs_threshold, 
                                      parallel=parallel,
                                      uncs_multi_mask=npy_loader['multi_uncs_mask'], 
                                      ens_pred=ens_seg)
            
            # evaluation full
            metrics_row = evaluate(gt, seg_filt, args.IoU_threshold, parallel)
            metrics_row['hash'] = filename_or_obj
            metrics_f = metrics_f.append(metrics_row, ignore_index=True)
            metrics_f.to_csv(os.path.join(args.path_save, f"filtered_metrics_{args.les_unc_measure}.csv"))

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
