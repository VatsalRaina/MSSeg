import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from joblib import Parallel, delayed
from functools import partial
import os


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


def plot_retention_curve(gts, preds, dsc_norm_scores, fracs_retained, n_jobs=None, save_path=None):
    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate((preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)[0]

    N = len(gts)

    uncs_ideal = np.absolute(gts - preds)
    ordering = uncs_ideal.argsort()
    gts_ideal = gts[ordering].copy()
    preds_ideal = preds[ordering].copy()

    uncs_random = np.linspace(0, 1000, len(gts))
    np.random.seed(0)
    np.random.shuffle(uncs_random)
    ordering = uncs_random.argsort()
    gts_random = gts[ordering].copy()
    preds_random = preds[ordering].copy()

    with Parallel(n_jobs=n_jobs) as parallel:
        process = partial(compute_dice_norm, preds_=preds_ideal, gts_=gts_ideal, N_=N)
        dsc_scores_ideal = np.asarray(
            parallel(delayed(process)(frac) for frac in fracs_retained)
        )
        process = partial(compute_dice_norm, preds_=preds_random, gts_=gts_random, N_=N)
        dsc_scores_random = np.asarray(
            parallel(delayed(process)(frac) for frac in fracs_retained)
        )

    plt.plot(fracs_retained, dsc_norm_scores, label="Uncertainty")
    plt.plot(fracs_retained, dsc_scores_ideal, label="Ideal")
    plt.plot(fracs_retained, dsc_scores_random, label="Random")
    plt.xlabel("Retention Fraction")
    plt.ylabel("DSC")
    plt.xlim([0.0, 1.01])
    plt.legend()
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "unc_ret.jpg")
        print(f"Saving figure at {save_path}")
    plt.savefig(save_path)
    plt.clf()


def get_dsc_norm_auc(gts, preds, uncs, n_jobs=None, plot=True, save_path=None):
    """ Get DSC_norm-AUC
    Parameters:
        - gts () - ground truth
        - preds () - predicted labels
        - uncs () - uncertainty metrics evaluation
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
        plot_retention_curve(gts, preds, dsc_norm_scores, fracs_retained, n_jobs=n_jobs, save_path=save_path)

    return metrics.auc(fracs_retained, dsc_norm_scores)
