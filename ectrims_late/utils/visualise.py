#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:59:22 2022

@author: Nataliia Molchanova
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import partial
import os
from .metrics import *


def plot_retention_curve_single(gts, preds, dsc_norm_scores, fracs_retained, n_jobs=None, save_path=None, unc_name=None):
    """
    For the precomputed normalized dice score for different retention fractions for a single subject.
    Called by metrics.get_dsc_norm_auc.
    """
    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds_ if pos == N_ else np.concatenate((preds_[:pos], gts_[pos:]))
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
    plt.xlabel(f"Retention Fraction ({unc_name})")
    plt.ylabel("DSC")
    plt.xlim([0.0, 1.01])
    plt.legend()
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "unc_ret.jpg")
        print(f"Saving figure at {save_path}")
    plt.savefig(save_path)
    plt.clf()


def plot_iqr_median_rc(metric_norm, fracs_ret, save_path):
    """
    Given the data frame with col: fracs_ret and index - scan plot the IQR, median retention curve
    """
    q1 = metric_norm.quantile(0.25)
    q2 = metric_norm.median()
    q3 = metric_norm.quantile(0.75)
    plt.plot(fracs_ret, q2, '-k', label='median')

    plt.fill_between(fracs_ret, q1, q3, color='gray', alpha=0.2, label='IQR')
    plt.xlim([0, 1.01])
    plt.xlabel("Retention fraction")
    plt.ylabel("F1 lesion-scale")
    plt.legend()
    plt.savefig(save_path)
    plt.clf()
    
def plot_mean_rc(metric_norm, fracs_ret, save_path):
    """
    Given the data frame with col: fracs_ret and index - scan plot the IQR, median retention curve
    """
    mean = metric_norm.mean()
    plt.plot(fracs_ret, mean, '-k')

    plt.xlim([0, 1.01])
    plt.legend()
    plt.xlabel("Retention fraction")
    plt.ylabel("F1 lesion-scale")
    plt.savefig(save_path)
    plt.clf()
