#!/bin/sh
#SBATCH --job-name insider_seed2
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --mail-user=nataliia.molchanova@chuv.ch
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/na2996/Desktop/logs
#SBATCH -o out/%x_%N.%j.%a.out
#SBATCH -e err/%x_%N.%j.%a.err

# scp -r /home/meri/data/insider/dataset na2996@hpc1.chuv.ch:/data/bach/MultipleSclerosis/nataliia/MSSeg/data/insider/

source activate msseg
cd /data/bach/MultipleSclerosis/nataliia/MSSeg/ectrims

srun python /data/bach/MultipleSclerosis/nataliia/MSSeg/MSSeg/ectrims/Training_cl_masking.py \
--learning_rate 5e-4 \
--n_epochs 200 \
--seed 2 \
--threshold 0.4 \
--path_train /data/bach/MultipleSclerosis/nataliia/MSSeg/data/insider/dataset/train \
--path_val /data/bach/MultipleSclerosis/nataliia/MSSeg/data/insider/dataset/val \
--flair_prefix FLAIR.nii.gz \
--mp2rage_prefix UNIT1.nii.gz \
--gts_prefix all_lesions.nii.gz \
--check_dataset \
--num_workers 8 \
--path_save /data/bach/MultipleSclerosis/nataliia/MSSeg/ectrims/cl_mask_training/seed2
