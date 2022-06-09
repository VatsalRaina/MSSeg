"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--path_load1', type=str, default='', help='Load path for numpy array')
parser.add_argument('--path_load2', type=str, default='', help='Load path for numpy array')
parser.add_argument('--path_load3', type=str, default='', help='Load path for numpy array')

def main(args):

    with open(args.path_load1, 'rb') as f:
        dsc_norm_scores1 = np.load(f)
        fracs_retained = np.load(f)

    with open(args.path_load2, 'rb') as f:
        dsc_norm_scores2 = np.load(f)

    with open(args.path_load3, 'rb') as f:
        dsc_norm_scores3 = np.load(f)

    plt.plot(fracs_retained, dsc_norm_scores1, label=r'$\text{Eval}_{\text{in}}$')
    plt.plot(fracs_retained, dsc_norm_scores2, label=r'$\text{Dev}_{\text{out}}$')
    plt.plot(fracs_retained, dsc_norm_scores3, label=r'$\text{Eval}_{\text{out}}$')
    plt.xlabel("Retention Fraction")
    plt.ylabel("nDSC")
    plt.xlim([0.0,1.01])
    plt.legend()
    plt.savefig('unc_ret_dsc_norm.png')
    plt.clf()

#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)