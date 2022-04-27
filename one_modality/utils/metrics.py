import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from joblib import Parallel, delayed
from functools import partial
import os
from scipy import ndimage
from scipy.interpolate import interp1d
from visualise import plot_retention_curve_single

from ..Uncertainty import lesions_uncertainty_sum
from transforms import get_FN_lesions_mask, get_TP_lesions_mask



def dice_metric(ground_truth, predictions):
    """
    Returns Dice coefficient for a single example.
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [W, H, D].
    Returns:
      Dice coefficient overlap (`float` in [0.0, 1.0])
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    ground_truth = ground_truth.astype("float32")
    predictions = predictions.astype("float32")

    # Calculate intersection and union of y_true and y_predict
    intersection = np.sum(predictions * ground_truth)
    union = np.sum(predictions) + np.sum(ground_truth)

    # Calcualte dice metric
    if intersection == 0.0 and union == 0.0:
        dice = 1.0
    else:
        dice = (2. * intersection) / union

    return dice


def dice_norm_metric(ground_truth, predictions):
    """
    For a single example returns DSC_norm, fpr, fnr
    """

    # Reference for normalized DSC
    r = 0.001
    # Cast to float32 type
    gt = ground_truth.astype("float32")
    seg = predictions.astype("float32")
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0, 1.0, 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = k * fp
        dsc_norm = 2 * tp / (fp_scaled + 2 * tp + fn)

        fpr = fp / (len(gt.flatten()) - np.sum(gt))
        if np.sum(gt) == 0:
            fnr = 1.0
        else:
            fnr = fn / np.sum(gt)
        return dsc_norm, fpr, fnr


def intersection_over_union(mask1, mask2):
    """Compute IoU for 2 binary masks
    mask1 and mask2 should have same dimensions
    """
    return np.sum(mask1 * mask2) / np.sum(mask1 + mask2 - mask1 * mask2)


def f1_lesion_metric(ground_truth, predictions, IoU_threshold):
    """
    For a single example returns DSC_norm, fpr, fnr

    Parameters:
        * ground_truth (numpy.ndarray) - size [H, W, D]
        * predictions (numpy.ndarray) - size [H, W, D]
        * IoU_threshold (float) - see description in the scientific report
    """

    tp, fp, fn = 0, 0, 0

    mask_multi_gt = ndimage.label(ground_truth)[0]
    mask_multi_pred = ndimage.label(predictions)[0]

    for label_pred in np.unique(mask_multi_pred):
        if label_pred != 0.0:
            mask_label_pred = (mask_multi_pred == label_pred).astype(int)
            all_iou = [0.0]
            # find maximum non-zero IoU of the connected component in the prediction with the gt
            for int_label_gt in np.unique(mask_multi_gt * mask_label_pred):  # iterate only intersections
                if int_label_gt != 0.0:
                    mask_label_gt = (mask_multi_gt == int_label_gt).astype(int)
                    all_iou.append(intersection_over_union(mask_label_pred, mask_label_gt))
            max_iou = max(all_iou)
            if max_iou >= IoU_threshold:
                tp += 1
            else:
                fp += 1

    # del mask_label_gt, mask_label_pred

    for label_gt in np.unique(mask_multi_gt):
        if label_gt != 0.0:
            mask_label_gt = (mask_multi_gt == label_gt).astype(int)
            all_iou = [0]
            for int_label_pred in np.unique(mask_multi_pred * mask_label_gt):
                if int_label_pred != 0.0:
                    mask_label_pred = (mask_multi_pred == int_label_pred).astype(int)
                    all_iou.append(intersection_over_union(mask_label_pred, mask_label_gt))
            max_iou = max(all_iou)
            if max_iou < IoU_threshold: fn += 1

    if tp + 0.5 * (fp + fn) == 0.0:
        return 0
    return tp / (tp + 0.5 * (fp + fn))


def get_dsc_norm(gts, preds, uncs, n_jobs=None):
    """ Get DSC_norm-AUC
    
    Parameters:
        - gts (1D numpy.ndarray) - ground truth
        - preds (1D numpy.ndarray) - predicted labels
        - uncs (1D numpy.ndarray) - uncertainty metrics evaluation
        - plot (bool) - if True, plots retention curve
        - n_jobs (int|None) - number of paraller jobs, joblib parameter.
        - save_path (path|None) - path to figure to be saved.
    Returns:
        area under retention curve (dice norm vs retention fraction)
    """

    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate((preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)[0]

    ordering = uncs.argsort()
    gts = gts[ordering]
    preds = preds[ordering]
    N = len(gts)

    # # Significant class imbalance means it is important to use logspacing between values
    # # so that it is more granular for the higher retention fractions
    fracs_retained = np.log(np.arange(200 + 1)[1:])
    fracs_retained /= np.amax(fracs_retained)

    process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N)
    dsc_norm_scores = np.asarray(
        Parallel(n_jobs=n_jobs)(delayed(process)(frac) for frac in fracs_retained)
    )

    return dsc_norm_scores


def get_dsc_norm_auc(gts, preds, uncs, n_jobs=None, plot=True, save_path=None):
    """ Get DSC_norm-AUC
    
    Parameters:
        - gts (1D numpy.ndarray) - ground truth
        - preds (1D numpy.ndarray) - predicted labels
        - uncs (1D numpy.ndarray) - uncertainty metrics evaluation
        - plot (bool) - if True, plots retention curve
        - n_jobs (int|None) - number of paraller jobs, joblib parameter.
        - save_path (path|None) - path to figure to be saved.
    Returns:
        area under retention curve (dice norm vs retention fraction)
    """

    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate((preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)[0]

    ordering = uncs.argsort()
    gts = gts[ordering]
    preds = preds[ordering]
    N = len(gts)

    # # Significant class imbalance means it is important to use logspacing between values
    # # so that it is more granular for the higher retention fractions
    fracs_retained = np.log(np.arange(200 + 1)[1:])
    fracs_retained /= np.amax(fracs_retained)

    process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N)
    dsc_norm_scores = np.asarray(
        Parallel(n_jobs=n_jobs)(delayed(process)(frac) for frac in fracs_retained)
    )

    if plot:
        plot_retention_curve_single(gts, preds, dsc_norm_scores, fracs_retained, n_jobs=n_jobs, save_path=save_path)

    return metrics.auc(fracs_retained, dsc_norm_scores)


def get_metric_for_rc(gts, preds, uncs, metric_name, fracs_retained, n_jobs=None):
    def compute_dice_norm(frac_, preds_, gts_, N_, ind_ret):
        pos = int(N_ * frac_)
        curr_preds = preds_ if pos == N_ else np.concatenate((preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)[ind_ret]

    def compute_dice(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds_ if pos == N_ else np.concatenate((preds_[:pos], gts_[pos:]))
        return dice_metric(gts_, curr_preds)

    ordering = uncs.argsort()
    gts = gts[ordering]
    preds = preds[ordering]
    N = len(gts)

    if metric_name == 'dsc_norm':
        process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N, ind_ret=0)
    elif metric_name == 'fpr':
        process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N, ind_ret=1)
    elif metric_name == 'fnr':
        process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N, ind_ret=2)
    elif metric_name == 'dice':
        process = partial(compute_dice, preds_=preds, gts_=gts, N_=N)
    else:
        raise NotImplementedError

    return Parallel(n_jobs=n_jobs)(delayed(process)(frac) for frac in fracs_retained)


def get_metric_for_rc_lesion(gts, preds, uncs, IoU_threshold, fracs_retained):
    """
    algorithm:
    0. obtain fn_lesions and tp_lesions from gt
    1. obtain tp, fp (form preds) and fn (from segm) lesions uncertainties and OHE masks.
    2. sort uncertainies and in the same order OHE masks
    3. for i_l in lesions_count:
        - retain lesion i_l and all lesions before it:
            if lesion is TP: remove from gt and pred
            if lesion is FP: remove from pred
            if lesion is FN: remove from ft
        - compute F1 with modified gt and seg
    :type gts: numpy.ndarray [H, W, D]
    :type preds: numpy.ndarray [H, W, D]
    :type uncs: numpy.ndarray [H, W, D]
    :type IoU_threshold: float
    :type fracs_retained: numpy.ndarray [n_fr,]
    :return:
    :rtype:
    """

    def retain_one_lesion(lesion_, pred, gt, IoU_threshold_):
        in_pred = True if np.sum(lesion_ * pred) == np.sum(lesion_) else False  # lesion is in prediction map
        in_gt = True if np.sum(lesion_ * gt) == np.sum(lesion_) else False  # lesion is in gt
        in_both = in_pred * in_gt  # fast track for perfect prediction

        if in_both:
            pred -= lesion_
            gt -= lesion_
            return pred, gt
        elif in_pred:
            pred -= lesion_  # remove if it is tp or fp
        elif in_gt:
            gt -= lesion_  # remove if it is fn
        else:
            raise NotImplementedError("Lesion not found")

        # for tp also remove from gt corresponding lesion
        if in_pred:
            mask_multi_gt = ndimage.label(gt)[0]
            max_iou = 0.0
            max_label = None
            for int_label_gt in np.unique(mask_multi_gt * lesion_):  # iterate only intersections
                mask_label_gt = (mask_multi_gt == int_label_gt).astype(int)
                iou = intersection_over_union(lesion_, mask_label_gt)
                if iou > max_iou:
                    max_iou = iou
                    max_label = int_label_gt
            if max_iou >= IoU_threshold_:
                gt -= (mask_multi_gt == max_label).astype(int)
        return pred, gt

    # compute masks and uncertainties
    fn_lesions = get_FN_lesions_mask(ground_truth=gts,
                                     predictions=preds,
                                     IoU_threshold=IoU_threshold,
                                     mask_type='binary')

    cc_uncs_fn, cc_mask_fn = lesions_uncertainty_sum(uncs_mask=uncs,
                                                     binary_mask=fn_lesions,
                                                     one_hot=True)
    cc_mask_fn = cc_mask_fn.astype("int")

    cc_uncs_pred, cc_mask_pred = lesions_uncertainty_sum(uncs_mask=uncs,
                                                         binary_mask=preds,
                                                         one_hot=True)
    cc_mask_pred = cc_mask_pred.astype("int")

    uncs_all = np.concatenate([cc_uncs_pred, cc_uncs_fn], axis=0)       # lesions uncertainties
    cc_mask_all = np.concatenate([cc_mask_pred, cc_mask_fn], axis=0)    # one hot encoded lesions masks [n_lesions, H, W, D]

    # sort uncertainties and lesions
    ordering = uncs_all.argsort()
    cc_mask_all = cc_mask_all[ordering]

    preds_ = preds.copy().astype("int")
    gts_ = gts.copy().astype("int")

    f1_values = []
    for i_l, lesion in enumerate(cc_mask_all[::-1]):        # exclude the most uncertain lesion
        preds_, gts_ = retain_one_lesion(lesion_=lesion, pred=preds_, gt=gts_, IoU_threshold_=IoU_threshold)
        f1_values.append(f1_lesion_metric(ground_truth=gts_, predictions=preds_, IoU_threshold=IoU_threshold))

    # interpolate the curve and make predictions in the retention fraction nodes
    n_lesions = cc_mask_all.shape[0]
    spline3_interpolator = interp1d(x=[_/n_lesions for _ in range(n_lesions)], y=f1_values,
                                    kind='cubic', fill_value="extrapolate")
    return spline3_interpolator(fracs_retained)
