#!/bin/bash

#SBATCH --gres=gpu:a40:4        # Request 4 A40 GPUs
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G       ## this is memory per node
#SBATCH --nodes=1
#SBATCH --ntasks=4                    # Total number of tasks (one per GPU)
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/output_%j.log        # Standard output and error log (%j expands to jobId)

# Load Conda environment
source /gscratch/cse/vsahil/miniconda/etc/profile.d/conda.sh
conda activate cleanclip-env

# Run the Python script
# srun python -m src.slurm_main --project_name clip-cc6m-pre-training --name trying_slurm --train_data ../CC12M/training_data/clean_data_cc6m.csv --image_key image --caption_key caption --distributed --batch_size 128 --num_workers 60 --lr 1e-3 --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random --patch_location random --patch_size 16 --eval_both_accuracy_and_asr --wandb --epochs 2
srun python -m src.main --project_name clip-cc6m-pre-training --name trying_slurm --train_data ../CC12M/training_data/clean_data_cc6m.csv --image_key image --caption_key caption --distributed --batch_size 128 --num_workers 60 --lr 1e-3 --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random --patch_location random --patch_size 16 --eval_both_accuracy_and_asr --wandb --epochs 2 --slurm_gpus
