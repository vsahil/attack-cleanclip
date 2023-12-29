'''
Code taken from CleanCLIP repository: https://github.com/nishadsinghi/CleanCLIP
'''

import os
os.environ["WANDB_API_KEY"] = "f17cbba930bd4473ba209b2a8f4ed8e244f8aece"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import sys, os, copy
import gc
import time
import wandb
import torch
import logging
import warnings
import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from pkgs.openai.clip import load as load_model

from .train import train
from .evaluate import evaluate
from .data import load as load_data
from .parser import parse_args
from .scheduler import cosine_scheduler
from .logger import get_logger, set_logger

mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")


def worker(rank, options, logger):
    if options.slurm_gpus:
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        torch.cuda.set_device(local_rank)
        options.device = f"cuda:{local_rank}"
    
        options.master = local_rank == 0
        options.rank = local_rank
        set_logger(rank = local_rank, logger = logger, distributed = options.distributed)
    else:
        options.rank = rank
        options.master = rank == 0
        
        set_logger(rank = rank, logger = logger, distributed = options.distributed)

        if(options.device == "cuda"):
            options.device += ":" + str(options.device_ids[options.rank] if options.distributed else options.device_id)

    logging.info(f"Using {options.device} device")

    if(options.master):
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if(options.distributed):
        dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
    
    options.batch_size = options.batch_size // options.num_devices

    model, processor = load_model(name = options.model_name, pretrained = options.pretrained)
    if options.shrink_and_perturb:
        random_init_model_copy = copy.deepcopy(model)

    if(options.device == "cpu"):
        model.float()
    else:
        if not options.slurm_gpus:
            torch.cuda.set_device(options.device_ids[options.rank] if options.distributed else options.device_id)
        model.to(options.device)
        if(options.distributed):
            model = DDP(model, device_ids = [options.device_ids[options.rank]])
        print("model loaded in rank ", options.rank)
    
    data = load_data(options, processor)

    if(options.master): print("DATA LOADED")
    
    optimizer = None
    scheduler = None
    linear_layer_deep_clustering_cheating_experiment = None
    if(data["train"] is not None):
        weight_decay_parameters = []
        no_weight_decay_parameters = []

        for name, parameter in model.named_parameters():
            if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                weight_decay_parameters.append(parameter)
                
            if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                no_weight_decay_parameters.append(parameter)
        
        if options.deep_clustering_cheating_experiment:
            import torch.nn as nn
            ## Also initialize a learnable linear layer at the start of each epoch. The input will be the image embeddings and the output will be the logits.
            linear_layer_deep_clustering_cheating_experiment = nn.Linear(1024, 1000).to(options.device)
            linear_layer_deep_clustering_cheating_experiment.weight.data.normal_(mean=0.0, std=0.01)
            linear_layer_deep_clustering_cheating_experiment.bias.data.zero_()
            ## set the requires_grad to True for the linear layer parameters.
            linear_layer_deep_clustering_cheating_experiment.weight.requires_grad = True
            linear_layer_deep_clustering_cheating_experiment.bias.requires_grad = True
            # optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": options.weight_decay} , {"params": linear_layer_deep_clustering_cheating_experiment.parameters(), "weight_decay": 0}], lr = options.lr, betas = (options.beta1, options.beta2), eps = options.eps)
        # else:
        optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": options.weight_decay}], lr = options.lr, betas = (options.beta1, options.beta2), eps = options.eps)
        scheduler = cosine_scheduler(optimizer, options.lr, options.num_warmup_steps, data["train"].num_batches * options.epochs)

    start_epoch = 0
    # import ipdb; ipdb.set_trace()
    ## we will automatically check if the last epoch checkpoint exists, and load it if it does. 
    if(options.checkpoint is not None):
        if options.checkpoint == "last":
            options.checkpoint = os.path.join(options.log_dir_path, "checkpoints", "epoch.last.pt")
        if(os.path.isfile(options.checkpoint)):
            checkpoint = torch.load(options.checkpoint, map_location = options.device)
            start_epoch = 0 if (options.complete_finetune or options.complete_finetune_save) else checkpoint['epoch'] if "epoch" in checkpoint else 0
            if options.eval_data_type in ["MSCOCO"]: start_epoch = 0        ## we are finetuning the model on retrieval datasets, but we also want to save the models, hence not doing complete_finetune, in which we do not save models. 
            state_dict = checkpoint["state_dict"]
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"Loaded checkpoint {options.checkpoint}")     # (start epoch {checkpoint['epoch']}
            del checkpoint, state_dict
            torch.cuda.empty_cache()
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")
            # raise Exception(f"No checkpoint found at {options.checkpoint}") -- there will be no exception is there is no epoch.last.pt file. 

    if options.deep_clustering_cheating_experiment:
        optimizer.add_param_group({"params": linear_layer_deep_clustering_cheating_experiment.parameters(), "weight_decay": 0})

    if options.shrink_and_perturb:
        ## We will load the weight, and add the lambda for this, and specific sigma for this.
        random_init_model_copy.to(options.device)
        params1 = random_init_model_copy.parameters()
        params2 = model.parameters()
        for p1, p2 in zip(*[params1, params2]):
            p2.data = copy.deepcopy(options.shrink_lambda * p2.data + options.perturb_lambda * p1.data)
        del random_init_model_copy
        ## here model will have the new weights. - please check - done

    cudnn.benchmark = True
    cudnn.deterministic = False
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    cudnn.allow_tf32 = True


    if(options.wandb and options.master):
        logging.debug("Starting wandb")
        project_name = options.project_name
        wandb.init(project = project_name, notes = options.notes, tags = [], config = vars(options))
        wandb.run.name = options.name
        wandb.save(os.path.join(options.log_dir_path, "params.txt"))

    # if(options.master):
        ## print the train and validation batch sizes. Use the dataloader
        # logging.info(f"Train Batch Size: {data['train'].batch_size}")
        # logging.info(f"Validation Batch Size: {data['validation'].batch_size}")
    # import ipdb; ipdb.set_trace()
    save_checkpoint = 1
    if options.eval_data_type in ["MSCOCO"] or options.epochs == 0 or options.shrink_and_perturb:
        save_checkpoint = 2
        evaluate(start_epoch, model, optimizer, processor, data, options)       ## This should give same results as zeroshot retrieval. We do not do this when zeroshot accuracy is the main target. 
    torch.cuda.empty_cache()
    
    if(data["train"] is not None):
        options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(options.checkpoints_dir_path, exist_ok = True)

        scaler = GradScaler()

        best_loss = np.inf
        for epoch in range(start_epoch + 1, options.epochs + 1):
            if(options.master): 
                logging.info(f"Starting Epoch {epoch}")

            start = time.time()
            train(epoch, model, data, optimizer, scheduler, scaler, options, processor, linear_layer_deep_clustering_cheating_experiment)
            end = time.time()

            if(options.master): 
                logging.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")

            metrics = evaluate(epoch, model, optimizer, processor, data, options)

            if(options.master) and not options.complete_finetune:       ## don't save checkpoints for the cleaning process. 
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                if epoch % save_checkpoint == 0:
                    torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))      # we don't need to save the model every epoch, just the best and latest one. 
                    ## also save this as the last checkpoint
                    torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.last.pt"))        ## this will be replaced at the end of each epoch.
                if("loss" in metrics):
                    if(metrics["loss"] < best_loss):
                        best_loss = metrics["loss"]
                        torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.best.pt"))

    if(options.distributed):
        dist.destroy_process_group()

    if(options.wandb and options.master):
        wandb.finish()


if(__name__ == "__main__"):
    os.environ["OMP_NUM_THREADS"] = "1" 
    options = parse_args()

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    if options.slurm_gpus:
        print("Using SLURM_GPUS")
        ngpus = int(os.environ.get('SLURM_GPUS', 1))     # Default to 1 if not set
    else:
        ngpus = torch.cuda.device_count()
    
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                assert not options.slurm_gpus, "Cannot use both SLURM_GPUS and device_ids"
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs = options.num_devices, args = (options, logger))
    
    listener.stop()
