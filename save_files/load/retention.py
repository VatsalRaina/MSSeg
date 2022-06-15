import os
import argparse
import re
from glob import glob
from joblib import Parallel, delayed
from functools import partial
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from uncertainty import ensemble_uncertainties_classification


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--threshold', type=float, default=0.35, help='Threshold for lesion detection')
parser.add_argument('--num_models', type=int, default=5, help='Number of models in ensemble')
parser.add_argument('--path_data', type=str, default='', help='Specify the path to the directory of np files')
parser.add_argument('--path_save', type=str, default='', help='Save path for retention curves')


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
            k = (1-r) * np.sum(gt) / ( r * ( len(gt.flatten()) - np.sum(gt) ) )
        tp = np.sum(seg[gt==1])
        fp = np.sum(seg[gt==0])
        fn = np.sum(gt[seg==0])
        fp_scaled = k * fp
        dsc_norm = 2 * tp / (fp_scaled + 2 * tp + fn)

        fpr = fp / ( len(gt.flatten()) - np.sum(gt) )
        if np.sum(gt) == 0:
            fnr = 1.0
        else:
            fnr = fn / np.sum(gt)
        return dsc_norm, fpr, fnr

def get_unc_score(gts, preds, uncs, n_jobs=4):

    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate((preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)[0]

    ordering = uncs.argsort()
    uncs = uncs[ordering]
    gts = gts[ordering]
    preds = preds[ordering]
    N = len(gts)
    
    # # Significant class imbalance means it is important to use logspacing between values
    # # so that it is more granular for the higher retention fractions
    num_values = 200
    fracs_retained = np.log(np.arange(num_values+1)[1:])
    fracs_retained = fracs_retained / np.amax(fracs_retained)

    process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N)
    dsc_norm_scores = np.asarray(
        Parallel(n_jobs=n_jobs)(delayed(process)(frac) for frac in fracs_retained)
    )

    return fracs_retained, dsc_norm_scores

def main(args):
    
    # Load the gts and predictions
    gt_loc = args.path_data + 'gt/'
    gt_paths = sorted(glob(os.path.join(gt_loc, "*.npy")), key=lambda i: int(re.sub('\D', '', i)))
    all_gts = []
    for gt_path in gt_paths:
        all_gts.append(np.load(gt_path))

    K = args.num_models
    all_model_preds = {}
    for i in range(K):
        all_model_preds[i+1] = []
        model_preds_loc = args.path_data + 'model' + str(i+1) + '/'
        model_preds_paths = sorted(glob(os.path.join(model_preds_loc, "*.npy")), key=lambda i: int(re.sub('\D', '', i)))
        for model_pred_path in model_preds_paths:
            all_model_preds[i+1].append(np.load(model_pred_path))

    P = len(all_gts)

    th = args.threshold

    all_curves_dsc_norm = {
        'confidence': [],
        'entropy_of_expected': [],
        'expected_entropy': [],
        'mutual_information': [],
        'epkl': [],
        'reverse_mutual_information': [],
        'ideal': [],
        'random': []
    }

    for p in range(P):
        print("Patient:", p)
        gt = all_gts[p]
        all_outputs = []
        for model_num in range(1,K+1):
            all_outputs.append(all_model_preds[model_num][p])
        all_outputs = np.asarray(all_outputs)
        outputs = np.mean(all_outputs, axis=0)

        uncs = ensemble_uncertainties_classification( np.concatenate( (np.expand_dims(all_outputs, axis=-1), np.expand_dims(1.-all_outputs, axis=-1)), axis=-1) )

        outputs[outputs>th]=1
        outputs[outputs<th]=0
        seg= np.squeeze(outputs)

        """
        Remove connected components smaller than 10 voxels
        """
        l_min = 9
        labeled_seg, num_labels = ndimage.label(seg)
        label_list = np.unique(labeled_seg)
        num_elements_by_lesion = ndimage.labeled_comprehension(seg,labeled_seg,label_list,np.sum,float, 0)

        seg2 = np.zeros_like(seg)
        for l in range(len(num_elements_by_lesion)):
            if num_elements_by_lesion[l] > l_min:
        # assign voxels to output
                current_voxels = np.stack(np.where(labeled_seg == l), axis=1)
                seg2[current_voxels[:, 0],
                    current_voxels[:, 1],
                    current_voxels[:, 2]] = 1
        seg=np.copy(seg2) 

        # Calculate all AUC-DSCs
        for unc_key, curr_uncs in uncs.items():
            fracs_retained, dsc_norm_curve = get_unc_score(gt.flatten(), seg.flatten(), curr_uncs.flatten(), n_jobs=args.n_jobs)
            all_curves_dsc_norm[unc_key].append(dsc_norm_curve)

    for unc_key, unc_curves_dsc_norm in all_curves_dsc_norm.items():
        scores = np.asarray(unc_curves_dsc_norm)
        scores = np.mean(unc_curves_dsc_norm, axis=0)
        all_curves_dsc_norm[unc_key] = scores

    # Plot the retention curves now
    for unc_key, scores in all_curves_dsc_norm.items():
        plt.plot(fracs_retained, scores, label=unc_key)
    plt.xlabel("Retention Fraction")
    plt.ylabel("nDSC")
    plt.xlim([0.0,1.01])
    plt.legend()
    plt.savefig(args.path_save)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)