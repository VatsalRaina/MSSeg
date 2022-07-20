#!/bin/bash

### DONE ############################################################################
python lesion_scale_retention_curve_new_npy.py \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_uncs_comp_npy \
--path_save /home/nataliia/results/combo_rc_f1025 \
--n_jobs 15 \
--IoU_threshold 0.25

# get lesion data
python lesion_scale_metrics_new_npy.py \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_uncs_comp_npy \
--path_save /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025 \
--n_jobs 20 \
--IoU_threshold 0.25

python lesion_scale_metrics_new_npy.py \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/lausanne_npz \
--path_save /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/lausanne_res_f1025 \
--n_jobs 20 \
--IoU_threshold 0.25 

# ideal and random rc
python lesion_scale_retention_idealrandom_curve_new_npy.py \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_uncs_comp_npy \
--path_save /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025

# threshold tunning
python lesion_scale_fp_thresh_tune_new_npy.py \
--les_unc_metric logsum \
--path_les_csv /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025/lesion_uncertainty_metrics_reverse_mutual_information.csv \
--path_fn_csv /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025/combo_uncs_values_npy/fn_count.csv \
--path_save /home/nataliia/results/combo_rc_f1025 \
--n_jobs 20

python lesion_scale_fp_thresh_tune_new_npy.py \
--les_unc_metric mean_iou_det \
--path_les_csv /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025/lesion_uncertainty_metrics_reverse_mutual_information.csv \
--path_fn_csv /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025/fn_count.csv \
--path_save /home/nataliia/results/combo_rc_f1025 \
--n_jobs 20

python lesion_scale_eval_thresh_npy.py \
--threshold 0.35 \
--les_unc_measure logsum \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/lausanne_npz \
--path_save /home/nataliia/results/lausanne_eval_f1025 \
--n_jobs 20 \
--IoU_threshold 0.25 \
--les_uncs_threshold 7.98664718097447 \
--eval_initial

python lesion_scale_eval_thresh_npy.py \
--threshold 0.35 \
--les_unc_measure mean_iou_det \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/lausanne_npz \
--path_save /home/nataliia/results/lausanne_eval_f1025 \
--n_jobs 20 \
--IoU_threshold 0.25 \
--les_uncs_threshold 0.494201680672269

####################################################################################

python lesion_scale_eval_thresh_npy.py \
--threshold 0.35 \
--les_unc_measure logsum \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/lausanne_npz \
--path_save /home/nataliia/results/lausanne_eval_f1025_gtlmin9 \
--n_jobs 20 \
--IoU_threshold 0.25 \
--les_uncs_threshold 7.98664718097447 \
--eval_initial \
&& \
python lesion_scale_eval_thresh_npy.py \
--threshold 0.35 \
--les_unc_measure mean_iou_det \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/lausanne_npz \
--path_save /home/nataliia/results/lausanne_eval_f1025_gtlmin9 \
--n_jobs 20 \
--IoU_threshold 0.25 \
--les_uncs_threshold 0.494201680672269

################################################################################################

# Filtering with all measures

python lesion_scale_fp_thresh_tune_new_npy.py \
--les_unc_metric mean \
--path_les_csv /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025/lesion_uncertainty_metrics_reverse_mutual_information.csv \
--path_fn_csv /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025/fn_count.csv \
--path_save /home/nataliia/results/combo_rc_f1025 \
--n_jobs 20 \
&& \
python lesion_scale_fp_thresh_tune_new_npy.py \
--les_unc_metric mean_ext \
--path_les_csv /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025/lesion_uncertainty_metrics_reverse_mutual_information.csv \
--path_fn_csv /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025/fn_count.csv \
--path_save /home/nataliia/results/combo_rc_f1025 \
--n_jobs 20 \
&& \
python lesion_scale_fp_thresh_tune_new_npy.py \
--les_unc_metric logsum_ext \
--path_les_csv /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025/lesion_uncertainty_metrics_reverse_mutual_information.csv \
--path_fn_csv /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_rc_f1025/fn_count.csv \
--path_save /home/nataliia/results/combo_rc_f1025 \
--n_jobs 20

python lesion_scale_eval_thresh_npy.py \
--threshold 0.35 \
--les_unc_measure mean \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/lausanne_npz \
--path_save /home/nataliia/results/lausanne_eval_f1025 \
--n_jobs 20 \
--IoU_threshold 0.25 \
--les_uncs_threshold 0.907923084497452 \
&& \
python lesion_scale_eval_thresh_npy.py \
--threshold 0.35 \
--les_unc_measure mean_ext \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/lausanne_npz \
--path_save /home/nataliia/results/lausanne_eval_f1025 \
--n_jobs 20 \
--IoU_threshold 0.25 \
--les_uncs_threshold 0.622932018628614 \
&& \
python lesion_scale_eval_thresh_npy.py \
--threshold 0.35 \
--les_unc_measure logsum_ext \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/lausanne_npz \
--path_save /home/nataliia/results/lausanne_eval_f1025 \
--n_jobs 20 \
--IoU_threshold 0.25 \
--les_uncs_threshold 9.43363935813626

##########################################################################################

# Recompute retention curves without FN

python lesion_scale_retention_curve_nofn_npy.py \
--path_data /mnt/nas4/datasets/ToCurate/MSSeg_canonical_nataliia/combo_uncs_comp_npy \
--path_save /home/nataliia/results/combo_rc_f1025_nofn \
--n_jobs 20 \
--IoU_threshold 0.25

python lesion_scale_retention_idealrandom_curve_new_npy.py \
--path_data /home/meri/uncertainty_challenge/unsure/faster_results/f1_025/combo_rc_f1025/combo_rc_f1025/lesions_data \
--path_save /home/meri/uncertainty_challenge/unsure/faster_results/f1_025/combo_rc_f0125_nofn