import numpy as np
from sklearn import metrics
from joblib import Parallel, delayed
from functools import partial
from scipy import ndimage
from scipy.interpolate import interp1d
from collections import Counter


def check_binary_mask(mask):
    unique = np.unique(mask)
    if np.sum(np.isin(unique, test_elements=[0.0, 1.0], invert=True)) != 0.0:
        raise ValueError(f"Expecting mask to be binary, got unique values: {unique}")

def ltpr(seg, gt):
    """
    get number of predicted connected components that have intersection with 
    the ground truth normalized by the amount of lesions on the ground truth
    ltpr = tp / (tp + fn) lesions
    """
    labeled_gt, num_labels = ndimage.label(gt)
    intersection = np.unique(seg * labeled_gt)
    return intersection[intersection != 0.0].size / num_labels if num_labels > 0.0 else 1.0

def lfpr(seg, gt, other_mask=None):
    """
    get number of predicted connected components that have no intersection 
    neither with gt nor with the other_mask normalized by the number of lesions
    on the prediction, i.e. fraction of fp lesions on the prediction
    """
    labeled_seg, num_labels = ndimage.label(seg)
    
    labels_seg = np.unique(labeled_seg)
    labels_seg = set(labels_seg[labels_seg != 0.0])
        
    intersection = np.unique(labeled_seg * gt)
    intersection = set(intersection[intersection != 0.0])
    
    if other_mask is not None:
        other_intersection = np.unique(labeled_seg * other_mask)
        other_intersection = set(other_intersection[other_intersection != 0.0])
    else:
        other_intersection = set()
    
    fp_labels = labels_seg - intersection - other_intersection
    
    return len(fp_labels) / num_labels if num_labels > 0.0 else 0.0
    
    
def tpr_fpr(ground_truth, predictions):
    _ground_truth = ground_truth.copy().flatten()
    _predictions = predictions.copy().flatten()
    
    fp = np.sum(_predictions[_ground_truth == 0])
    tp = np.sum(_predictions[_ground_truth == 1])
    
    return tp / np.sum(_ground_truth == 1), fp / np.sum(_ground_truth == 0), tp / (tp + fp)


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
            # if np.sum(mask_label_pred) > n_min_vox:
            all_iou = [0.0]
            # find maximum non-zero IoU of the connected component in the prediction with the gt
            # iterate only intersections
            for int_label_gt in np.unique(mask_multi_gt * mask_label_pred):
                if int_label_gt != 0.0:
                    mask_label_gt = (mask_multi_gt == int_label_gt).astype(int)
                    all_iou.append(intersection_over_union(
                        mask_label_pred, mask_label_gt))
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
                    mask_label_pred = (mask_multi_pred ==
                                       int_label_pred).astype(int)
                    all_iou.append(intersection_over_union(
                        mask_label_pred, mask_label_gt))
            max_iou = max(all_iou)
            if max_iou < IoU_threshold:
                fn += 1

    if tp + 0.5 * (fp + fn) == 0.0:
        return 1.
    return tp / (tp + 0.5 * (fp + fn))

def lesion_metrics_parallel(ground_truth, predictions, other_mask, IoU_threshold, parallel_backend):
    """
    For a single example returns DSC_norm, fpr, fnr

    Parameters:
        * ground_truth (numpy.ndarray) - size [H, W, D]
        * predictions (numpy.ndarray) - size [H, W, D]
        * IoU_threshold (float) - see description in the scientific report
    """
    def get_tp_fp(mask_label_pred, mask_multi_gt, mask_multi_other):
        # mask_label_pred = (mask_multi_pred == label_pred).astype(int)
        all_iou = [0.0]
        # iterate only intersections
        for int_label_gt in np.unique(mask_multi_gt * mask_label_pred):
            if int_label_gt != 0.0:
                mask_label_gt = (mask_multi_gt == int_label_gt).astype(int)
                all_iou.append(intersection_over_union(
                    mask_label_pred, mask_label_gt))
        max_iou = max(all_iou)
        if max_iou >= IoU_threshold:
            return 'tp'
        else:
            if mask_multi_other is not None:
                other_iou = [0.0]
                for other_label in np.unique(mask_multi_other * mask_label_pred):
                    if other_label != 0.0:
                        mask_label_other = (mask_multi_other == other_label).astype(int)
                        other_iou.append(intersection_over_union(
                            mask_label_pred, mask_label_other))
                if max(other_iou) < IoU_threshold:
                    return 'fp'
                else:
                    return 'other'
            else:
                return 'fp'

    def get_fn(mask_label_gt, mask_multi_pred):
        # mask_label_gt = (mask_multi_gt == label_gt).astype(int)
        all_iou = [0.0]
        for int_label_pred in np.unique(mask_multi_pred * mask_label_gt):
            if int_label_pred != 0.0:
                mask_label_pred = (mask_multi_pred == int_label_pred).astype(int)
                all_iou.append(intersection_over_union(
                    mask_label_pred, mask_label_gt))
        max_iou = max(all_iou)
        if max_iou < IoU_threshold:
            return 1
        else:
            return 0

    mask_multi_pred_, n_les_pred = ndimage.label(predictions)
    mask_multi_gt_, n_les_gt = ndimage.label(ground_truth)
    mask_multi_other_, n_les_other = ndimage.label(other_mask)

    process_fp_tp = partial(get_tp_fp,
                            mask_multi_gt=mask_multi_gt_,
                            mask_multi_other=mask_multi_other_)

    tp_fp = parallel_backend(delayed(process_fp_tp)((mask_multi_pred_ == label_pred).astype(int))
                             for label_pred in np.unique(mask_multi_pred_) 
                             if label_pred != 0)
    counter = Counter(tp_fp)
    tp = float(counter['tp'])
    fp = float(counter['fp'])

    process_fn = partial(get_fn, mask_multi_pred=mask_multi_pred_)

    fn = parallel_backend(delayed(process_fn)((mask_multi_gt_ == label_gt).astype(int))
                          for label_gt in np.unique(mask_multi_gt_) 
                          if label_gt != 0)
    fn = float(np.sum(fn))
    
    tpr = tp / (tp + fn) if tp + fn > 0 else 1.0
    fnr = 1 - tpr
    fpr = fp / n_les_pred if n_les_pred > 0 else 0.0
    ppv = tp / (tp + fp) if tp + fp > 0 else 1.0
    f1 = 1.0 if tp + 0.5 * (fp + fn) == 0.0 else tp / (tp + 0.5 * (fp + fn))

    return f1, tpr, fnr, fpr, ppv

def f1_lesion_metric_parallel(ground_truth, predictions, IoU_threshold, parallel_backend):
    """
    For a single example returns DSC_norm, fpr, fnr

    Parameters:
        * ground_truth (numpy.ndarray) - size [H, W, D]
        * predictions (numpy.ndarray) - size [H, W, D]
        * IoU_threshold (float) - see description in the scientific report
    """

    def get_tp_fp(label_pred, mask_multi_pred, mask_multi_gt):
        mask_label_pred = (mask_multi_pred == label_pred).astype(int)
        all_iou = [0.0]
        # iterate only intersections
        for int_label_gt in np.unique(mask_multi_gt * mask_label_pred):
            if int_label_gt != 0.0:
                mask_label_gt = (mask_multi_gt == int_label_gt).astype(int)
                all_iou.append(intersection_over_union(
                    mask_label_pred, mask_label_gt))
        max_iou = max(all_iou)
        if max_iou >= IoU_threshold:
            return 'tp'
        else:
            return 'fp'

    def get_fn(label_gt, mask_multi_pred, mask_multi_gt):
        mask_label_gt = (mask_multi_gt == label_gt).astype(int)
        all_iou = [0]
        for int_label_pred in np.unique(mask_multi_pred * mask_label_gt):
            if int_label_pred != 0.0:
                mask_label_pred = (mask_multi_pred ==
                                   int_label_pred).astype(int)
                all_iou.append(intersection_over_union(
                    mask_label_pred, mask_label_gt))
        max_iou = max(all_iou)
        if max_iou < IoU_threshold:
            return 1
        else:
            return 0

    mask_multi_pred_, n_les_pred = ndimage.label(predictions)
    mask_multi_gt_, n_les_gt = ndimage.label(ground_truth)

    process_fp_tp = partial(get_tp_fp, mask_multi_pred=mask_multi_pred_,
                            mask_multi_gt=mask_multi_gt_)

    tp_fp = parallel_backend(delayed(process_fp_tp)(label_pred)
                             for label_pred in np.unique(mask_multi_pred_) if label_pred != 0)
    counter = Counter(tp_fp)
    tp = float(counter['tp'])
    fp = float(counter['fp'])

    process_fn = partial(get_fn, mask_multi_pred=mask_multi_pred_,
                         mask_multi_gt=mask_multi_gt_)

    fn = parallel_backend(delayed(process_fn)(label_gt)
                          for label_gt in np.unique(mask_multi_gt_) if label_gt != 0)
    fn = float(np.sum(fn))
    
    tpr = tp / (tp + fn) if tp + fn > 0 else 1.0
    fnr = 1 - tpr
    fpr = fp / n_les_pred if n_les_pred > 0 else 0.0
    ppv = tp / (tp + fp) if tp + fp > 0 else 1.0
    f1 = 1.0 if tp + 0.5 * (fp + fn) == 0.0 else tp / (tp + 0.5 * (fp + fn))

    return f1, tpr, fnr, fpr, ppv


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
        curr_preds = preds if pos == N_ else np.concatenate(
            (preds_[:pos], gts_[pos:]))
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
        Parallel(n_jobs=n_jobs)(delayed(process)(frac)
                                for frac in fracs_retained)
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
        curr_preds = preds if pos == N_ else np.concatenate(
            (preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)[0]

    from .visualise import plot_retention_curve_single

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
        Parallel(n_jobs=n_jobs)(delayed(process)(frac)
                                for frac in fracs_retained)
    )

    if plot:
        plot_retention_curve_single(
            gts, preds, dsc_norm_scores, fracs_retained, n_jobs=n_jobs, save_path=save_path)

    return metrics.auc(fracs_retained, dsc_norm_scores)


def get_metric_for_rc(gts, preds, uncs, metric_name, fracs_retained, n_jobs=None):
    def compute_dice_norm(frac_, preds_, gts_, N_, ind_ret):
        pos = int(N_ * frac_)
        curr_preds = preds_ if pos == N_ else np.concatenate(
            (preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)[ind_ret]

    def compute_dice(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds_ if pos == N_ else np.concatenate(
            (preds_[:pos], gts_[pos:]))
        return dice_metric(gts_, curr_preds)

    ordering = uncs.argsort()
    gts = gts[ordering]
    preds = preds[ordering]
    N = len(gts)

    if metric_name == 'dsc_norm':
        process = partial(compute_dice_norm, preds_=preds,
                          gts_=gts, N_=N, ind_ret=0)
    elif metric_name == 'fpr':
        process = partial(compute_dice_norm, preds_=preds,
                          gts_=gts, N_=N, ind_ret=1)
    elif metric_name == 'fnr':
        process = partial(compute_dice_norm, preds_=preds,
                          gts_=gts, N_=N, ind_ret=2)
    elif metric_name == 'dice':
        process = partial(compute_dice, preds_=preds, gts_=gts, N_=N)
    else:
        raise NotImplementedError

    return Parallel(n_jobs=n_jobs)(delayed(process)(frac) for frac in fracs_retained)



def get_metric_for_rc_lesion(gts, preds, uncs, IoU_threshold, fracs_retained, n_jobs):
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

    from .uncertainty import lesions_uncertainty_sum
    from .transforms import get_FN_lesions_mask_parallel, get_TP_FP_lesions_mask_parallel

    uncs_list = [None for _ in range(3)]  # fn, tp, fp

    with Parallel(n_jobs=n_jobs) as parallel_backend:
        # compute masks and uncertainties
        lesions = get_FN_lesions_mask_parallel(ground_truth=gts,
                                               predictions=preds,
                                               IoU_threshold=IoU_threshold,
                                               mask_type='binary',
                                               parallel_backend=parallel_backend)
        uncs_list[0] = lesions_uncertainty_sum(uncs_mask=uncs,
                                               binary_mask=lesions,
                                               dtype=int,
                                               mask_type='one_hot')[0]

        lesions = get_TP_FP_lesions_mask_parallel(ground_truth=gts,
                                                  predictions=preds,
                                                  IoU_threshold=IoU_threshold,
                                                  mask_type='binary',
                                                  parallel_backend=parallel_backend)

        uncs_list[1] = lesions_uncertainty_sum(uncs_mask=uncs,
                                               binary_mask=lesions[0],
                                               dtype=int,
                                               mask_type='one_hot')[0]

        uncs_list[2] = lesions_uncertainty_sum(uncs_mask=uncs,
                                               binary_mask=lesions[1],
                                               dtype=int,
                                               mask_type='one_hot')[0]

        lesion_type_list = [np.full(shape=unc.shape, fill_value=fill)
                            for unc, fill in zip(uncs_list, ['fn', 'tp', 'fp'])
                            if unc.size > 0]
        uncs_list = [unc for unc in uncs_list if unc.size > 0]

        uncs_all = np.concatenate(uncs_list, axis=0)
        lesion_type_all = np.concatenate(lesion_type_list, axis=0)

        del uncs_list, lesion_type_list, lesions
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
    spline_interpolator = interp1d(x=[_ / n_lesions for _ in range(n_lesions + 1)], y=f1_values[::-1],
                                   kind='slinear', fill_value="extrapolate")
    return spline_interpolator(fracs_retained), f1_values[::-1]