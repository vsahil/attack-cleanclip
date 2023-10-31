import subprocess

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


import random, os
import subprocess
import time

def run_expts():
    processes = []
    models = ['mmcl', 'mmcl_ssl']
    cleaning_approaches = ['mmcl']

    datasets = ['cc3m', 'cc6m']
    project_name = "cleanclip-retrieval-finetuning"

    for model in models:
        for approach in cleaning_approaches:
            if "ssl" in approach:
                batch_size = 64
                lrs = [1e-4, 5e-4, 1e-3, 5e-3]
            else:
                batch_size = 128
                lrs = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 5e-5, 1e-4]
            
            # for weight_decay_values in weight_decays:
            for dataset in datasets:
                poisoned_examples = 3000 if dataset == 'cc6m' else 1500 if dataset == 'cc3m' else None
                for lr in lrs:
                    expt_name = f'finetuning-mscoco_karpathy-train_pretrained_dataset_{dataset}_and_objective_{model}_lr_{lr}'

                    if dataset == 'cc6m':
                        if model == 'mmcl':       ## experiments with CC6M
                            checkpoint = 'logs/train_cc6m_poison_mmcl_1e_3/checkpoints/epoch_21.pt'
                        elif model == 'mmcl_ssl':
                            checkpoint = 'logs/train_cc6m_poison_mmcl_ssl_1e_3_batch1024/checkpoints/epoch_36.pt'
                        else:
                            raise NotImplementedError
                    elif dataset == 'cc3m':
                        if model == 'mmcl':       ## experiments with CC6M
                            checkpoint = 'logs/train_1_poison_mmcl/checkpoints/epoch.best.pt'
                        elif model == 'mmcl_ssl':
                            checkpoint = 'logs/train_newa100_poison_mmcl_ssl_both_1e_3_lr/checkpoints/epoch.best.pt'
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                    
                    # Get the list of available GPUs
                    available_gpus = get_available_gpus()
                    ## If there are no available GPUs, wait and try again later
                    while not available_gpus:
                        time.sleep(100)
                        available_gpus = get_available_gpus()
                    device_id = available_gpus[0]       # Use the first available GPU
                    
                    # device_id = 0
                    
                    if approach == 'mmcl':
                        extra = ''
                    elif approach == 'ssl':
                        extra = '--inmodal --clip_weight 0'
                    elif approach == 'mmcl_ssl':
                        extra = '--inmodal --clip_weight 1'
                    
                    port = random.randint(100, 6000)

                    command = f"time python -m src.main --project_name {project_name} --name {expt_name} --checkpoint {checkpoint} --train_data utils/data/MSCOCO/mscoco_train.csv --validation_data utils/data/MSCOCO/mscoco_val.csv --eval_test_data_dir utils/data/MSCOCO/mscoco_test.csv --eval_data_type MSCOCO --add_backdoor --asr --patch_type random  --patch_location random --patch_size 16 --image_key image --caption_key caption --device_id {device_id} --batch_size {batch_size} --num_workers 10 --wandb --epochs 21 --num_warmup_steps 50 --lr {lr} --distributed_init_method 'tcp://127.0.0.1:{port}' "                

                    # command = f"time python -m src.main --name {expt_name} --checkpoint {checkpoint} --train_data ../CC12M/training_data/clean_data_cc6m.csv --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random  --patch_location random --patch_size 16 --image_key image --caption_key caption --device_id {device_id} --batch_size {batch_size} --num_workers 10 --wandb --epochs 10 --num_warmup_steps 50 --lr {lr} --complete_finetune {extra} --eval_both_accuracy_and_asr --weight_decay {weight_decay_values} --distributed_init_method 'tcp://127.0.0.1:{port}'  "            ## 100K cleaning data

                    # command = f"time python -m src.main --name {expt_name} --checkpoint {checkpoint} --train_data ../CC12M/second_training_data/clean_data_cc6m_200k.csv --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random  --patch_location random --patch_size 16 --image_key image --caption_key caption --device_id {device_id} --batch_size {batch_size} --num_workers 10 --wandb --epochs 10 --num_warmup_steps 50 --lr {lr} --complete_finetune {extra} --eval_both_accuracy_and_asr --distributed_init_method 'tcp://127.0.0.1:{port}' "          ## 200K cleaning data
                    
                    print(command, "\n")
                    process = subprocess.Popen(command, shell=True)
                    processes.append(process)
                    time.sleep(120)
        
    for process in processes:
        process.wait()     

if __name__ == '__main__':
    # print(get_available_gpus())
    run_expts()
