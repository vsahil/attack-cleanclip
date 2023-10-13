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
    models = ['mmcl_ssl'] #, 'mmcl']
    cleaning_approaches = ['ssl']       # ['mmcl', 'mmcl_ssl', ]
    # lrs = [1e-5, 4e-5, 8e-5, 1e-4, 4e-4, 8e-4, 1e-3, 4e-3]      ## 8 lrs
    # lrs = [1e-7, 3e-7, 7e-7, 1e-6, 3e-6, 7e-6, 1e-5, 3e-5]      ## 8 lrs
    # lrs = [1e-9, 5e-9, 1e-8, 5e-8]      ## because we reduced the batch size for ssl and mmcl_ssl models, so we also experiment with smaller learning rates.

    poisoned_examples = 3000

    for model in models:
        for approach in cleaning_approaches:
            if "ssl" in approach:
                batch_size = 64
                lrs = [1e-4] #, 3e-4, 1e-3, 3e-3]
                # lrs = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 3e-7, 7e-7, 1e-6, 3e-6, 7e-6, 1e-5, 3e-5]
            else:
                batch_size = 128
                lrs = [1e-4, 3e-4, 1e-3, 3e-3]        # 
                # lrs = [3e-6, 7e-6, 1e-5, 3e-5]  # [1e-7, 3e-7, 7e-7, 1e-6] #
            for lr in lrs:
                if poisoned_examples == 5000 or poisoned_examples == 3000:
                    expt_name = f'cleaning_poisoned_cc6m_{model}_{poisoned_examples}poison_clean_{approach}_lr_{lr}'
                elif poisoned_examples == 1500:
                    expt_name = f'cleaning_poisoned_cc6m_{model}_poison_clean_{approach}_lr_{lr}'
                else:   raise NotImplementedError
                
                if model == 'mmcl':       ## experiments with CC3M
                    checkpoint = 'logs/train_cc6m_poison_mmcl_1e_3/checkpoints/epoch_21.pt'
                elif model == 'mmcl_ssl':
                    checkpoint = 'logs/train_cc6m_poison_mmcl_ssl_1e_3_batch1024/checkpoints/epoch_36.pt'
                
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
                command = f"time python -m src.main --name {expt_name} --checkpoint {checkpoint} --train_data ../CC12M/training_data/clean_data_cc6m.csv --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random  --patch_location random --patch_size 16 --image_key image --caption_key caption --device_id {device_id} --batch_size {batch_size} --num_workers 10 --wandb --epochs 20 --num_warmup_steps 50 --lr {lr} --complete_finetune {extra} --eval_both_accuracy_and_asr --distributed_init_method 'tcp://127.0.0.1:{port}' "
                # command = f"time python -m src.main --name {expt_name} --checkpoint {checkpoint} --train_data ../CC12M/second_training_data/clean_data_cc6m_200k.csv --eval_test_data_dir data/ImageNet1K/validation/ --eval_data_type ImageNet1K --add_backdoor --asr --patch_type random  --patch_location random --patch_size 16 --image_key image --caption_key caption --device_id {device_id} --batch_size {batch_size} --num_workers 10 --wandb --epochs 20 --num_warmup_steps 50 --lr {lr} --complete_finetune {extra} --eval_both_accuracy_and_asr --distributed_init_method 'tcp://127.0.0.1:{port}' "
                print(command, "\n")
                # os.system(command)
                process = subprocess.Popen(command, shell=True)
                # processes.append(process)
                # Wait for a minute to allow the GPU to get filled
                time.sleep(120)
    
    for process in processes:
        process.wait()     

if __name__ == '__main__':
    # print(get_available_gpus())
    run_expts()
