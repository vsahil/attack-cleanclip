import subprocess
import random
import time


def get_available_gpus():
    """
    Returns a list of GPU IDs that are currently not in use.
    """
    try:
        # Run the nvidia-smi command and get the output
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv'], check=True, text=True, capture_output=True)
        output = result.stdout

        # Parse the output to get the memory usage of each GPU
        lines = output.strip().split('\n')[1:] # Skip the header line
        gpu_memory_used = [int(line.split()[0]) for line in lines]

        # Find the GPUs that are not in use (memory usage is 0)
        available_gpus = [i for i, mem in enumerate(gpu_memory_used) if mem <= 10]

        return available_gpus

    except subprocess.CalledProcessError:
        print("Failed to run nvidia-smi.")
        return []

    except Exception as e:
        print("Error:", e)
        return []


def run_expts(sbatch_run):
    all_commands = []
    processes = []
    models = ['mmcl']   #, 'mmcl']
    cleaning_approaches = ['mmcl_ssl']      # 'mmcl', 'ssl', 
    h2lab_gpus = True

    # dataset = 'cc6m'
    # dataset = 'pretrained_cc6m'
    # dataset = 'cc6m_warped_poison'
    dataset = 'cc6m_label_consistent_poison'
    poisoned_examples = 3000 if dataset == 'cc6m' else 1500 if dataset == 'cc3m' else 1500 if dataset == 'pretrained_cc6m' else 3000 if dataset == 'cc6m_warped_poison' else 3000 if dataset == 'cc6m_label_consistent_poison' else None
    weight_decay_values = 0.1
    # weight_decays = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    ssl_weight_values = 1       ## This is the default value for the weight of the inmodal loss. Both mmcl and ssl have weight 1. 
    ssl_weights = [1]   #, 2, 4, 6, 8] #, 10, 12]
    # project_name = "clip-defense-cc6m-complete-finetune"
    # project_name = "clip-defense-pretrained-cc6m-complete-finetune"
    # project_name = "clip-defense-cc6m-complete-finetune-warped"
    project_name = "clip-defense-cc6m-complete-finetune-label-consistent"
    dummy_run = True

    for model in models:
        for approach in cleaning_approaches:
            if "ssl" in approach:
                batch_size = 64
            else:
                batch_size = 128
            
            # lrs = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5]
            # lrs = [6e-6, 7e-6, 8e-6, 9e-6, 1.1e-5, 1.2e-5, 1.3e-5, 1.4e-5]
            # lrs = [1.5e-5, 1.6e-5, 1.7e-5, 1.8e-5, 1.9e-5, 2e-5, 2.1e-5, 2.2e-5]
            # lrs = [2.3e-5, 2.4e-5, 2.5e-5, 2.6e-5, 2.7e-5, 2.8e-5, 2.9e-5, 3e-5]
            
            # lrs = [3.75e-5, 4e-5, 4.25e-5, 4.75e-5]
            # lrs = [3.1e-5, 3.2e-5, 3.3e-5, 3.4e-5, 3.5e-5, 3.6e-5, 3.7e-5, 3.8e-5]
            
            # lrs = [4.7e-5, 4.8e-5, 4.9e-5]
            # lrs = [3.9e-5, 4e-5, 4.1e-5, 4.2e-5, 4.3e-5, 4.4e-5, 4.5e-5, 4.6e-5]
            # lrs = [1.5e-5, 1.75e-5, 2e-5, 2.25e-5, 2.5e-5, 2.75e-5, 3e-5, 3.25e-5]
            
            # lrs = [6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4]
            # lrs = [5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 2e-3, 3e-3]
            # lrs = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3]
            lrs = [1.1e-4, 1.2e-4, 1.3e-4, 1.4e-4, 1.5e-5, 1.6e-4, 1.7e-4, 1.8e-4]
            
            # for weight_decay_values in weight_decays:
            for ssl_weight_values in ssl_weights:
                for lr in lrs:
                    if (poisoned_examples == 5000 or poisoned_examples == 3000) and dataset == 'cc6m':
                        # expt_name = f'cleaning_poisoned_cc6m_{model}_{poisoned_examples}poison_clean_{approach}_lr_{lr}_weight_decay_{weight_decay_values}'
                        expt_name = f'cleaning_poisoned_cc6m_{model}_{poisoned_examples}poison_clean_{approach}_lr_{lr}_ssl_weight_{ssl_weight_values}'
                    elif poisoned_examples == 1500 and dataset == 'cc6m':
                        expt_name = f'cleaning_poisoned_cc6m_{model}_poison_clean_{approach}_lr_{lr}'
                    elif poisoned_examples == 1500 and dataset == 'pretrained_cc6m':
                        expt_name = f'cleaning_poisoned_pretrained_cc6m_{model}_poison_clean_{approach}_lr_{lr}'
                    elif poisoned_examples == 3000 and dataset == 'cc6m_warped_poison':
                        expt_name = f'cleaning_poisoned_cc6m_{model}_poison_clean_{approach}_lr_{lr}'
                    elif poisoned_examples == 3000 and dataset == 'cc6m_label_consistent_poison':
                        expt_name = f'cleaning_poisoned_cc6m_{model}_poison_3000_label_consistent_clean_{approach}_lr_{lr}'
                    else:   
                        raise NotImplementedError
                    
                    if model == 'mmcl' and dataset == 'cc6m':
                        checkpoint = 'logs/train_cc6m_poison_mmcl_1e_3/checkpoints/epoch_21.pt'
                    elif model == 'mmcl_ssl' and dataset == 'cc6m':
                        checkpoint = 'logs/train_cc6m_poison_mmcl_ssl_1e_3_batch1024/checkpoints/epoch_36.pt'
                    elif model == 'mmcl' and dataset == 'pretrained_cc6m':
                        checkpoint = 'logs/train_cc6m_poison_pretrained_mmcl_5e_7_poison_3000/checkpoints/epoch_18.pt'
                    elif model == 'mmcl_ssl' and dataset == 'pretrained_cc6m':
                        checkpoint = 'logs/train_cc6m_poison_pretrained_mmcl_ssl_5e_7_poison_3000/checkpoints/epoch_13.pt'
                    elif model == 'mmcl' and dataset == 'cc6m_warped_poison':
                        checkpoint = "logs/train_cc6m_poison_mmcl_1e_3_poison_3000_warped/checkpoints/epoch_33.pt"
                    elif model == 'mmcl_ssl' and dataset == 'cc6m_warped_poison':
                        checkpoint = "logs/train_cc6m_poison_mmcl_ssl_1e_3_poison_3000_warped/checkpoints/epoch_43.pt"
                    elif model == "mmcl" and dataset == 'cc6m_label_consistent_poison':
                        checkpoint = "logs/train_cc6m_poison_mmcl_ssl_1e_3_poison_3000_label_consistent/checkpoints/epoch_37.pt"
                    elif model == "mmcl_ssl" and dataset == 'cc6m_label_consistent_poison':
                        checkpoint = "logs/train_cc6m_poison_mmcl_1e_3_poison_3000_label_consistent/checkpoints/epoch_40.pt"
                    else:
                        raise NotImplementedError
                    
                    if dummy_run:
                        device_id = 0
                    else:
                        ## Get the list of available GPUs
                        available_gpus = get_available_gpus()

                        ## If there are no available GPUs, wait and try again later
                        while not available_gpus:
                            time.sleep(100)
                            available_gpus = get_available_gpus()
                        # Use the first available GPU
                        device_id = available_gpus[0]
                    
                    if approach == 'mmcl':
                        extra = ''
                    elif approach == 'ssl':
                        extra = '--inmodal --clip_weight 0'
                    elif approach == 'mmcl_ssl':
                        extra = f'--inmodal --clip_weight 1 --inmodal_weight {ssl_weight_values}'
                        # if ssl_weight_values > 1:
                        # else:
                        #     extra = '--inmodal --clip_weight 1'
                    port = random.randint(100, 6000)

                    complete_finetune = "--complete_finetune" # if h2lab_gpus else "--complete_finetune_save" -- no need to save as checkpoint cannot be saved. 

                    command = f"time python -m src.main --project_name {project_name} --name {expt_name} --checkpoint {checkpoint} --train_data /gscratch/cse/vsahil/CC12M/training_data/clean_data_cc6m.csv --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random  --patch_location random --patch_size 16 --image_key image --caption_key caption --device_id {device_id} --batch_size {batch_size} --num_workers 8 --wandb --epochs 20 --num_warmup_steps 50 --lr {lr} {complete_finetune} {extra} --eval_both_accuracy_and_asr --weight_decay {weight_decay_values} --distributed_init_method 'tcp://127.0.0.1:{port}'  "            ## 100K cleaning data

                    # command = f"time python -m src.main --name {expt_name} --checkpoint {checkpoint} --train_data ../CC12M/second_training_data/clean_data_cc6m_200k.csv --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random  --patch_location random --patch_size 16 --image_key image --caption_key caption --device_id {device_id} --batch_size {batch_size} --num_workers 10 --wandb --epochs 10 --num_warmup_steps 50 --lr {lr} --complete_finetune {extra} --eval_both_accuracy_and_asr --distributed_init_method 'tcp://127.0.0.1:{port}' "          ## 200K cleaning data
                    
                    print(command, "\n")

                    if not dummy_run and not sbatch_run:
                        process = subprocess.Popen(command, shell=True)
                        processes.append(process)
                        time.sleep(120)     # # Wait for a minute to allow the GPU to get filled
                    
                    if sbatch_run:
                        all_commands.append(command)
        
    for process in processes:
        process.wait()
    
    if sbatch_run:
        if h2lab_gpus:
            for command in all_commands:
                sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=cleaning-label-consistent-expt
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vsahil@uw.edu
#SBATCH --output=slurm_outputs/output_%j.log

source /gscratch/cse/vsahil/miniconda/etc/profile.d/conda.sh
conda activate robustness_cleanclip
{command}
"""
                # Use subprocess to submit the job
                proc = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE, text=True)
                proc.communicate(sbatch_script)
                # print(sbatch_script)
                    
        else:       ## cse ckpt gpus. 
            for command in all_commands:
                sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=cleaning-expt
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vsahil@uw.edu
#SBATCH --output=slurm_outputs/output_%j.log

source /gscratch/cse/vsahil/miniconda/etc/profile.d/conda.sh
conda activate robustness_cleanclip
{command}
"""
                # Use subprocess to submit the job
                proc = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE, text=True)
                proc.communicate(sbatch_script)
                # print(sbatch_script)


if __name__ == '__main__':
    # print(get_available_gpus())
    run_expts(sbatch_run=True)

