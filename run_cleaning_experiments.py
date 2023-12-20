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
    models = ['mmcl_ssl']   #, 'mmcl']
    cleaning_approaches = ['mmcl_ssl']      # 'mmcl', 'ssl', 

    dataset = 'cc6m'
    poisoned_examples = 3000 if dataset == 'cc6m' else 1500 if dataset == 'cc3m' else None
    weight_decay_values = 0.1
    # weight_decays = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    ssl_weight_values = 1       ## This is the default value for the weight of the inmodal loss. Both mmcl and ssl have weight 1. 
    ssl_weights = [2, 4, 6, 8] #, 10, 12]
    project_name = "clip-defense-cc6m-complete-finetune"
    dummy_run = False

    for model in models:
        for approach in cleaning_approaches:
            if "ssl" in approach:
                batch_size = 64
                lrs = [5e-5, 1e-4, 5e-4, 1e-3]
            else:
                batch_size = 128
                lrs = [1e-4, 5e-4, 1e-3, 5e-3]
            
            # for weight_decay_values in weight_decays:
            for ssl_weight_values in ssl_weights:
                for lr in lrs:
                    if poisoned_examples == 5000 or poisoned_examples == 3000:
                        # expt_name = f'cleaning_poisoned_cc6m_{model}_{poisoned_examples}poison_clean_{approach}_lr_{lr}_weight_decay_{weight_decay_values}'
                        expt_name = f'cleaning_poisoned_cc6m_{model}_{poisoned_examples}poison_clean_{approach}_lr_{lr}_ssl_weight_{ssl_weight_values}'
                    elif poisoned_examples == 1500:
                        expt_name = f'cleaning_poisoned_cc6m_{model}_poison_clean_{approach}_lr_{lr}'
                    else:   raise NotImplementedError
                    
                    if model == 'mmcl':       ## experiments with CC3M
                        checkpoint = 'logs/train_cc6m_poison_mmcl_1e_3/checkpoints/epoch_21.pt'
                    elif model == 'mmcl_ssl':
                        checkpoint = 'logs/train_cc6m_poison_mmcl_ssl_1e_3_batch1024/checkpoints/epoch_36.pt'
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
                        if ssl_weight_values > 1:
                            extra = f'--inmodal --clip_weight 1 --inmodal_weight {ssl_weight_values}'
                        else:
                            extra = '--inmodal --clip_weight 1'
                    port = random.randint(100, 6000)


                    command = f"time python -m src.main --project_name {project_name} --name {expt_name} --checkpoint {checkpoint} --train_data ../CC12M/training_data/clean_data_cc6m.csv --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random  --patch_location random --patch_size 16 --image_key image --caption_key caption --device_id {device_id} --batch_size {batch_size} --num_workers 10 --wandb --epochs 20 --num_warmup_steps 50 --lr {lr} --complete_finetune {extra} --eval_both_accuracy_and_asr --weight_decay {weight_decay_values} --distributed_init_method 'tcp://127.0.0.1:{port}'  "            ## 100K cleaning data

                    # command = f"time python -m src.main --name {expt_name} --checkpoint {checkpoint} --train_data ../CC12M/second_training_data/clean_data_cc6m_200k.csv --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random  --patch_location random --patch_size 16 --image_key image --caption_key caption --device_id {device_id} --batch_size {batch_size} --num_workers 10 --wandb --epochs 10 --num_warmup_steps 50 --lr {lr} --complete_finetune {extra} --eval_both_accuracy_and_asr --distributed_init_method 'tcp://127.0.0.1:{port}' "          ## 200K cleaning data
                    
                    print(command, "\n")

                    if not dummy_run:
                        process = subprocess.Popen(command, shell=True)
                        processes.append(process)
                        time.sleep(120)     # # Wait for a minute to allow the GPU to get filled
        
    for process in processes:
        process.wait()     

if __name__ == '__main__':
    # print(get_available_gpus())
    run_expts()
