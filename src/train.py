'''
Code taken from CleanCLIP repository: https://github.com/nishadsinghi/CleanCLIP
'''

import time, os, copy
import wandb
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast


def get_loss(umodel, outputs, criterion, options, gather_backdoor_indices, kmeans=None, this_batch_cluster_labels=None, linear_layer=None):  
    assert (not options.defense) and (not options.unlearn)  ### I am not using options.defense, so this should be true.
    assert gather_backdoor_indices is None   ### I am not using options.defense, so this should be true. 
    
    if(options.inmodal):
        image_embeds, augmented_image_embeds = outputs.image_embeds[:len(outputs.image_embeds) // 2], outputs.image_embeds[len(outputs.image_embeds) // 2:]
        text_embeds, augmented_text_embeds = outputs.text_embeds[:len(outputs.text_embeds) // 2], outputs.text_embeds[len(outputs.text_embeds) // 2:]
    else:
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
            
    if(options.distributed):
        if(options.inmodal):         ## This is for SSL part of the code. 
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            augmented_gathered_image_embeds = [torch.zeros_like(augmented_image_embeds) for _ in range(options.num_devices)]
            augmented_gathered_text_embeds = [torch.zeros_like(augmented_text_embeds) for _ in range(options.num_devices)]
            
            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
            dist.all_gather(augmented_gathered_image_embeds, augmented_image_embeds)
            dist.all_gather(augmented_gathered_text_embeds, augmented_text_embeds)
            
            image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])
            augmented_image_embeds = torch.cat(augmented_gathered_image_embeds[:options.rank] + [augmented_image_embeds] + augmented_gathered_image_embeds[options.rank + 1:])
            augmented_text_embeds  = torch.cat(augmented_gathered_text_embeds[:options.rank]+ [augmented_text_embeds] + augmented_gathered_text_embeds[options.rank + 1:])
        else:       ## This is for MMCL part of the code, where there is no augmenations. So this will also work for SigLIP training. 
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]

            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)

            image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])
            
            if (gather_backdoor_indices is not None) and (not options.unlearn):
                normal_indices = (~torch.cat(gather_backdoor_indices)).nonzero().squeeze()
                image_embeds = image_embeds[normal_indices]
                text_embeds  = text_embeds[normal_indices]

    logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_image_per_text = logits_text_per_image.t()

    if(options.inmodal):
        logits_image_per_augmented_image = umodel.logit_scale.exp() * image_embeds @ augmented_image_embeds.t()
        logits_text_per_augmented_text = umodel.logit_scale.exp() * text_embeds @ augmented_text_embeds.t()

    if(options.deep_clustering or options.deep_clustering_cheating_experiment):
        ## now we apply a crossentropy loss between the logits for each image and the predicted psuedo label from the clustering process. critereon is nn.CrossEntropyLoss()
        cluster_label_predicted_logits = linear_layer(image_embeds)
        ## cluster psuedo labels
        
        if (options.deep_clustering_cheating_experiment):
            cluster_labels = torch.from_numpy(this_batch_cluster_labels).to(options.device, non_blocking = True)
        else:
            _, cluster_labels = kmeans.index.search(image_embeds.detach().cpu().numpy(), 1)
            cluster_labels = torch.from_numpy(cluster_labels).to(options.device, non_blocking = True)

        assert cluster_label_predicted_logits.shape[0] == cluster_labels.shape[0]
        ## compute the loss
        clustering_loss = criterion(cluster_label_predicted_logits, cluster_labels.squeeze())
         
    batch_size = len(logits_text_per_image)
    
    target = torch.arange(batch_size).long().to(options.device, non_blocking = True)
    
    if options.siglip:      ## this is when we are using binary cross entropy loss for training the two modalities
        def get_ground_truth_siglip(device, dtype, num_logits) -> torch.Tensor:
            labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
            return labels

        def siglip_loss(image_embeds):
            # logits = self.get_logits(image_features, text_features, logit_scale)  logits_text_per_image
            labels = get_ground_truth_siglip(image_embeds.device, image_embeds.dtype, image_embeds.shape[0])
            loss = -torch.nn.functional.logsigmoid(labels * logits_text_per_image).mean()   # / image_embeds.shape[0]
            return loss
        
        crossmodal_contrastive_loss = siglip_loss(image_embeds)
        contrastive_loss = options.siglip_weight * crossmodal_contrastive_loss
        # print("Siglip loss: ", crossmodal_contrastive_loss.mean().item(), contrastive_loss.mean().item(), options.siglip_weight)
    else:       ## This is the normal MMCL loss
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2     ## this is the MMCL loss
        contrastive_loss = (options.clip_weight * crossmodal_contrastive_loss)

    if (options.inmodal):
        inmodal_contrastive_loss = (criterion(logits_image_per_augmented_image, target) + criterion(logits_text_per_augmented_text, target)) / 2
        # contrastive_loss = (crossmodal_contrastive_loss + inmodal_contrastive_loss) / 2     ## This gives equal weightage to both the losses
        contrastive_loss += (options.inmodal_weight * inmodal_contrastive_loss)
        # print("total loss", contrastive_loss.mean().item(), inmodal_contrastive_loss.mean().item(), options.inmodal_weight)
    
    if (options.deep_clustering or options.deep_clustering_cheating_experiment):      ## deep clustering cheating experiment was wrong this now. 
        contrastive_loss += (options.deep_clustering_weight * clustering_loss)
    
    # if options.deep_clustering and options.inmodal:
    #     print("MMCL loss: ", crossmodal_contrastive_loss.mean().item(), "SSL loss: ", inmodal_contrastive_loss.mean().item(), "Deep clustering loss: ", clustering_loss.mean().item())
    # elif options.deep_clustering and not options.inmodal:
    #     print("MMCL loss: ", crossmodal_contrastive_loss.mean().item(), "Deep clustering loss: ", clustering_loss.mean().item())

    if (gather_backdoor_indices is not None) and (options.unlearn):
        normal_indices = (~torch.cat(gather_backdoor_indices)).nonzero()
        normal_indices = normal_indices[:,0] if len(normal_indices.shape) == 2 else normal_indices
        backdoor_indices = torch.cat(gather_backdoor_indices).nonzero()
        backdoor_indices = backdoor_indices[:,0] if len(backdoor_indices.shape) == 2 else backdoor_indices
        if len(normal_indices) and len(backdoor_indices):
            contrastive_loss = contrastive_loss[normal_indices].mean() - contrastive_loss[backdoor_indices].mean()
        elif len(normal_indices):
            contrastive_loss = contrastive_loss[normal_indices].mean()
        else:
            contrastive_loss = -contrastive_loss[backdoor_indices].mean()
    elif options.unlearn:
        contrastive_loss = contrastive_loss.mean()

    loss = contrastive_loss
    return loss, contrastive_loss, torch.tensor(0)


# @torch.no_grad()
# def get_clean_batch(model, batch, options, step, threshold = 0.6):
#     input_ids, attention_mask, pixel_values, pixel_values_cropped = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True), batch["pixel_values_cropped"].to(options.device, non_blocking = True)
#     pixel_values_all = torch.cat([pixel_values, pixel_values_cropped])
#     outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values_all)
#     image_embeds = outputs.image_embeds
#     image_embeds, image_embeds_cropped = image_embeds[: len(image_embeds) // 2], image_embeds[len(image_embeds) // 2 :] 
#     pairwise_similarity = 1 - (((image_embeds - image_embeds_cropped)**2).sum(dim = 1) / 2)
#     is_normal = pairwise_similarity > threshold ## if the pairwise similarity is high the it is an original image 
#     indices = is_normal.nonzero().squeeze()
#     # indices = range(len(pixel_values)) if len(indices) == 0 else indices ## don't want any empty batch

#     is_backdoor = batch["is_backdoor"].to(options.device, non_blocking = True)
#     total_backdoors = sum(is_backdoor).item()
#     predicted_backdoor = ~ is_normal  
#     fraction_caught = -1

#     if sum(predicted_backdoor).item() != len(predicted_backdoor): 
#         backdoor_predicted_equal = is_backdoor & predicted_backdoor
#         correct_backdoors = sum(backdoor_predicted_equal).item()
#         if total_backdoors > 0:
#             fraction_caught = correct_backdoors // total_backdoors

#     if options.wandb and options.master:
#         wandb.log({f'{options.rank}/len of indices' : len(indices), 'step': step})
#         wandb.log({f'{options.rank}/# images removed' : len(pixel_values) - len(indices), 'step': step})
#         wandb.log({f'{options.rank}/total backdoors' : total_backdoors, 'step': step})      
#         wandb.log({f'{options.rank}/correct backdoors detected' : correct_backdoors, 'step': step})      
#         wandb.log({f'{options.rank}/fraction of backdoors caught' : fraction_caught, 'step': step})      

    # return input_ids[indices], attention_mask[indices], pixel_values[indices], torch.tensor(len(indices)).to(options.device) 
    # return is_normal


def process_batch(model, batch, options, step):
    input_ids, attention_mask, pixel_values, is_backdoor = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True), batch["is_backdoor"].to(options.device, non_blocking = True)
    outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
    with torch.no_grad():
        similarity = torch.diagonal(outputs.image_embeds @ outputs.text_embeds.t())
        topmax     = int(options.remove_fraction * len(similarity))
        detect_indices = similarity.topk(topmax).indices
    num_backdoor = is_backdoor.sum().item()
    backdoor_indices = is_backdoor.nonzero()
    backdoor_indices = backdoor_indices[:,0] if len(backdoor_indices.shape) == 2 else backdoor_indices
    count = 0
    if len(backdoor_indices) > 0:
        for backdoor_index in backdoor_indices:
            count += (backdoor_index in detect_indices)
    if options.wandb and options.master:
        wandb.log({f'{options.rank}/total backdoors' : num_backdoor, 'step': step})      
        wandb.log({f'{options.rank}/correct backdoors detected' : count, 'step': step})   
    pred_backdoor_indices = torch.zeros_like(similarity).int()
    pred_backdoor_indices[detect_indices] = 1
    return outputs, pred_backdoor_indices


def train(epoch, model, data, optimizer, scheduler, scaler, options, processor_eval, linear_layer_deep_clustering_cheating_experiment=None):    
    dataloader = data["train"]
    if(options.distributed): dataloader.sampler.set_epoch(epoch)

    model.train()
    criterion = nn.CrossEntropyLoss().to(options.device) if not options.unlearn else nn.CrossEntropyLoss(reduction = 'none').to(options.device)

    modulo = max(1, int(dataloader.num_samples / options.batch_size / 10))
    umodel = model.module if(options.distributed) else model

    start = time.time()
    kmeans = None
    # linear_layer = None
    cluster_labels = None
    this_batch_cluster_labels = None

    # import ipdb; ipdb.set_trace()
    if(options.deep_clustering or options.deep_clustering_cheating_experiment):
        if options.deep_clustering_cheating_experiment:
            import pandas as pd
            cluster_labels = pd.read_csv(f"deep_clustering_cheating_experiment/cleaning_image_labels.csv", header=0)       ## this has the image name, idx, and label. 
            cluster_labels = cluster_labels["label"].to_numpy()
            assert len(cluster_labels) >= len(dataloader.dataset)       ## well this will be greater because dataloader has a drop_last=True
            assert len(set(cluster_labels)) <= 1000     ## 100 imageNet classes
            # if epoch == 1:      ## Initialize the linear layer only for the first epoch for the cheating experiment.
            #     ## Also initialize a learnable linear layer at the start of each epoch. The input will be the image embeddings and the output will be the logits.
            #     linear_layer = nn.Linear(1024, 1000).to(options.device)
            #     linear_layer.weight.data.normal_(mean=0.0, std=0.01)
            #     linear_layer.bias.data.zero_()
            #     ## set the requires_grad to True for the linear layer parameters.
            #     linear_layer.weight.requires_grad = True
            #     linear_layer.bias.requires_grad = True
            #     # optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": options.weight_decay}], lr = options.lr, betas = (options.beta1, options.beta2), eps = options.eps)
            #     optimizer.add_param_group({"params": linear_layer.parameters(), "weight_decay": 0})
            # else:
                ## use the linear layer parameters from the previous epoch. The linear layer is not added to the model ,hence we cannot do umodel.linear_layer or model.linear_layer
                # linear_layer = nn.Linear(1024, 1000).to(options.device) -
                # linear_layer.weight.data = copy.deepcopy(optimizer.param_groups[-1]["params"][0].data) -- this technique is not working . 
                # linear_layer.bias.data = copy.deepcopy(optimizer.param_groups[-1]["params"][1].data)
        else:
            ## we can increase the batch size of the dataloader but only doing it when we are not using distributed training.
            if not options.distributed:
                from torch.utils.data import DataLoader
                new_dataloader = DataLoader(dataloader.dataset, batch_size = 1024, shuffle=False, num_workers = options.num_workers, pin_memory=False, sampler=None, drop_last=True)
                new_dataloader.num_samples = len(new_dataloader) * 1024
                new_dataloader.num_batches = len(new_dataloader)
            else:
                new_dataloader = dataloader
            
            ## Before starting every epoch, we need to perform clustering. 
            ncentroids = 1000
            niter = 20
            verbose = False
            ## Get the image embeddings for the entire training dataset.
            if(options.master): print("BUILDING CLUSTERS AT START OF EPOCH ", epoch)
            x = []
            with torch.no_grad():       ## we can optimize this if we can just get the image embeddings. 
                for index, batch in enumerate(new_dataloader):
                    if options.inmodal:
                        pixel_values = batch["pixel_values"][0].to(options.device, non_blocking = True)
                    else:
                        pixel_values = batch["pixel_values"].to(options.device, non_blocking = True)
                    # outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
                    image_embeds = model.get_image_features(pixel_values)
                    image_embeds /= image_embeds.norm(dim = -1, keepdim = True)
                    image_embeds = image_embeds.cpu()
                    x.append(image_embeds)
                    del pixel_values
                x = torch.cat(x)
            # import ipdb; ipdb.set_trace()
            x = x.cpu().numpy()
            import faiss
            kmeans = faiss.Kmeans(x.shape[1], ncentroids, niter=niter, verbose=verbose, gpu=False, spherical=True)
            kmeans.train(x)
            assert x.shape[1] == 1024
            
            linear_layer = None
            ## Also initialize a learnable linear layer at the start of each epoch. The input will be the image embeddings and the output will be the logits.
            linear_layer = nn.Linear(1024, 1000).to(options.device)
            linear_layer.weight.data.normal_(mean=0.0, std=0.01)
            linear_layer.bias.data.zero_()
            linear_layer_deep_clustering_cheating_experiment = linear_layer
            optimizer.add_param_group({"params": linear_layer_deep_clustering_cheating_experiment.parameters(), "weight_decay": 0})
        
        if(options.master): print("CLUSTERING DONE")


    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")

    for index, batch in enumerate(dataloader):
        step = dataloader.num_batches * epoch + index
        scheduler(step)

        optimizer.zero_grad()
        
        if(options.inmodal):
            input_ids, attention_mask, pixel_values = batch["input_ids"][0].to(options.device, non_blocking = True), batch["attention_mask"][0].to(options.device, non_blocking = True), batch["pixel_values"][0].to(options.device, non_blocking = True)
            augmented_input_ids, augmented_attention_mask, augmented_pixel_values = batch["input_ids"][1].to(options.device, non_blocking = True), batch["attention_mask"][1].to(options.device, non_blocking = True), batch["pixel_values"][1].to(options.device, non_blocking = True)
            input_ids = torch.cat([input_ids, augmented_input_ids])
            attention_mask = torch.cat([attention_mask, augmented_attention_mask])
            pixel_values = torch.cat([pixel_values, augmented_pixel_values])
        else:
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True)
        
        gather_backdoor_indices = None
        if options.defense and epoch > options.defense_epoch:
            outputs, pred_backdoor_indices = process_batch(model, batch, options, step)
            gather_backdoor_indices = [torch.zeros_like(pred_backdoor_indices) for _ in range(options.num_devices)]
            dist.all_gather(tensor_list = gather_backdoor_indices, tensor = pred_backdoor_indices)
        else:
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)

        with autocast():
            if options.deep_clustering_cheating_experiment:
                ## select the cluster labels for the current batch use "original_idx" from the items to select the datapoints from the full_cluster_labels
                this_batch_cluster_labels = cluster_labels[batch["original_idx"].tolist()]
            loss, contrastive_loss, cyclic_loss = get_loss(umodel, outputs, criterion, options, gather_backdoor_indices, kmeans, this_batch_cluster_labels, linear_layer_deep_clustering_cheating_experiment)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        
        scaler.update()
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)
        ## the linear layer to the model, so that it can be used in the next epoch.

        end = time.time()
        # if (options.master and index % 2000 == 0 and index > 0):      ## we don't need this anymore, this was added to save model between epochs, they were surprisingly good, the saving is done in evaluate.py for these models.
        #     logging.info(f"Done with {index} data points")
        #     if options.complete_finetune:
        #         from .evaluate import evaluate
        #         print("Evaluating at step: ", step)
        #         evaluate(epoch, model, optimizer, processor_eval, data, options, step)

        if(options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
            num_samples = (index + 1) * len(input_ids) * options.num_devices
            dataloader_num_samples = dataloader.num_samples

            print(f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")

            # metrics = {"loss": loss.item(), "contrastive_loss": contrastive_loss.item(), "cyclic_loss": cyclic_loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
            metrics = {"train_loss": loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step})
        
            start = time.time()
    
    # if options.deep_clustering_cheating_experiment:
    #     import ipdb; ipdb.set_trace()
    #     print(linear_layer.weight.data)
        # optimizer.param_groups[-1]["params"][0].data = linear_layer.weight.data
        # optimizer.param_groups[-1]["params"][1].data = linear_layer.bias.data
        ## add the weight of the new linear_layer 
