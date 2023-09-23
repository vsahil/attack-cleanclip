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
    cleaning_approaches = ['mmcl', 'ssl', 'mmcl_ssl']
    # models = ['mmcl_ssl']
    # cleaning_approaches = ['mmcl']
    # lrs = [1e-5, 4e-5, 8e-5, 1e-4, 4e-4, 8e-4, 1e-3, 4e-3]      ## 8 lrs
    lrs = [1e-7, 3e-7, 7e-7, 1e-6, 3e-6, 7e-6, 1e-5, 3e-5]      ## 8 lrs

    poisoned_examples = 3000

    for model in models:
        for approach in cleaning_approaches:
            for lr in lrs:
                if poisoned_examples == 5000 or poisoned_examples == 3000:
                    expt_name = f'cleaning_poisoned_{model}_{poisoned_examples}poison_clean_{approach}_lr_{lr}'
                elif poisoned_examples == 1500:
                    expt_name = f'cleaning_poisoned_{model}_poison_clean_{approach}_lr_{lr}'
                else:   raise NotImplementedError
                
                if model == 'mmcl':       ## experiments with CC3M
                    checkpoint = 'logs/train_cc6m_poison_mmcl_1e_3/checkpoints/epoch_23.pt'
                elif model == 'mmcl_ssl':
                    checkpoint = 'logs/train_cc6m_poison_mmcl_ssl_1e_3_batch1024/checkpoints/epoch_27.pt'
                
                # if model == 'mmcl':    ## experiments with cleaning pretrained 400M models -- the bug was this was approach. That means several of them have bugs. 
                #     if poisoned_examples == 1500:
                #         checkpoint = 'logs/poisoned_pretrained_400m_with_mmcl_loss_lr_2e_6/checkpoints/epoch_4.pt'
                #     elif poisoned_examples == 5000:
                #         checkpoint = 'logs/poisoned_pretrained_400m_with_mmcl_loss_5000poison_lr_2e_6/checkpoints/step_13218.pt'
                
                # elif model == 'mmcl_ssl':
                #     if poisoned_examples == 1500:
                #         checkpoint = 'logs/poisoned_pretrained_400m_with_mmcl_ssl_loss_lr_4e_6/checkpoints/step_9312.pt'
                #     elif poisoned_examples == 5000:
                #         checkpoint = 'logs/poisoned_pretrained_400m_with_mmcl_ssl_loss_5000poison_lr_2e_6/checkpoints/step_9312.pt'
                
                # else:
                #     raise NotImplementedError
                
                # Get the list of available GPUs
                available_gpus = get_available_gpus()

                # If there are no available GPUs, wait and try again later
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
                    extra = '--inmodal --clip_weight 1'
                port = random.randint(100, 6000)
                command = f"time python -m src.main --name {expt_name} --checkpoint {checkpoint} --train_data ../CC12M/training_data/clean_banana_random_random_16_10000000_3000.csv --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random  --patch_location random --patch_size 16 --image_key image --caption_key caption --device_id {device_id} --batch_size 128 --num_workers 10 --wandb --epochs 20 --num_warmup_steps 50 --lr {lr} --complete_finetune {extra} --eval_both_accuracy_and_asr --distributed_init_method 'tcp://127.0.0.1:{port}' "
                print(command, "\n")
                # os.system(command)
                process = subprocess.Popen(command, shell=True)
                processes.append(process)
                # Wait for a minute to allow the GPU to get filled
                time.sleep(60)
    
    for process in processes:
        process.wait()     

if __name__ == '__main__':
    # print(get_available_gpus())
    run_expts()
