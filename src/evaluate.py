'''
Code taken from CleanCLIP repository: https://github.com/nishadsinghi/CleanCLIP
'''

import wandb, os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
from .scheduler import cosine_scheduler


def get_validation_metrics(model, dataloader, options):
    logging.info("Started validating")

    metrics = {}

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction = "sum").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if(options.inmodal):
                # item["input_ids"] = self.captions["input_ids"][idx], self.augment_captions["input_ids"][idx]
                # item["attention_mask"] = self.captions["attention_mask"][idx], self.augment_captions["attention_mask"][idx]
                # item["pixel_values"] = self.processor.process_image(image), self.processor.process_image(_augment_image(os.path.join(self.root, self.images[idx])))
                input_ids = batch['input_ids'][0].to(options.device, non_blocking = True)
                attention_mask = batch["attention_mask"][0].to(options.device, non_blocking = True)
                pixel_values = batch["pixel_values"][0].to(options.device, non_blocking = True)
            else:  
                input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True)
            # import ipdb; ipdb.set_trace()
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            
            umodel = model.module if(options.distributed) else model

            logits_per_image = umodel.logit_scale.exp() * outputs.image_embeds @ outputs.text_embeds.t()
            logits_per_text = logits_per_image.t()

            target = torch.arange(len(input_ids)).long().to(options.device, non_blocking = True)
            loss = (criterion(logits_per_image, target) + criterion(logits_per_text, target)) / 2

            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["val_loss"] = loss

    logging.info("Finished validating")

    return metrics


@torch.no_grad()
def get_zeroshot_metrics(model, processor, test_dataloader, options, do_asr, accuracy_on_poisoned_images=False, text_embeddings=None, image_embeddings=None, label_list=None):
    logging.info("Started zeroshot testing")

    model.eval()
    umodel = model.module if(options.distributed) else model


    config = eval(open(f"{options.eval_test_data_dir}/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]
    with torch.no_grad():
        if do_asr:
            backdoor_target_index = list(filter(lambda x: 'banana' in classes[x], range(len(classes))))
            backdoor_target_index = torch.tensor(backdoor_target_index[0]).to(options.device)
        
        if text_embeddings is None:
            text_embeddings = []
            for c in tqdm(classes):
                text = [template(c) for template in templates]
                text_tokens = processor.process_text(text)
                text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device) 
                text_embedding = umodel.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
                text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
                text_embedding = text_embedding.mean(dim = 0)
                text_embedding /= text_embedding.norm()
                text_embeddings.append(text_embedding)
            text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.device)

    if image_embeddings is None:
        image_embeddings = []
        label_list = []
        for image, label in tqdm(test_dataloader):
            image = image.to(options.device)
            image_embedding = umodel.get_image_features(image)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
            image_embeddings.append(image_embedding)
            label_list.append(label)
        image_embeddings = torch.cat(image_embeddings, dim = 0).to(options.device)
        label_list = torch.cat(label_list, dim = 0).to(options.device)
    
    with torch.no_grad():
        topk = [1, 5, 10]
        correct = {k: 0 for k in topk}
        total = 0
        
        # for image, label in tqdm(test_dataloader):
        #     image, label = image.to(options.device), label.to(options.device)
        #     image_embedding = umodel.get_image_features(image)
        #     image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
        #     logits = (image_embedding @ text_embeddings)
        #     ranks = logits.topk(max(topk), 1)[1].T

        logits = (image_embeddings @ text_embeddings)       ## can do in one go, as the image embeddings are already calculated.
        ranks = logits.topk(max(topk), 1)[1].T
        
        if do_asr:
            non_label_indices = (label_list != backdoor_target_index).nonzero().squeeze()       ## this removes the indices of the test set that are not the backdoor target class
            if type(non_label_indices) == int or len(non_label_indices) > 0:
                ranks = ranks[:, non_label_indices]
            predictions = ranks == backdoor_target_index
        else:
            predictions = ranks == label_list
        total += predictions.shape[1]
        for k in topk:
            correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item()

    if do_asr:
        results = {f"asr_top{k}": correct[k] / total for k in topk}
    elif accuracy_on_poisoned_images:
        results = {f"zeroshot_top{k}_images_with_trigger": correct[k] / total for k in topk}
    else:
        results = {f"zeroshot_top{k}": correct[k] / total for k in topk}
    logging.info("Finished zeroshot testing")

    return results, text_embeddings


@torch.no_grad()
def get_zeroshot_retrieval_metrics(model, processor, test_dataloader, options):
    logging.info("Started zeroshot retrieval")
    model.eval()
    image_embeddings = []
    text_embeddings = []
    ## unlike the zeroshot retrieval we did earlier, we cannot store embeddings of the images and texts as the model is being trained.
    for index, batch in enumerate(test_dataloader):     ## The images already have a trigger as we are adding backdoors to them. 
        input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True)        
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
        image_embeddings.append(outputs.image_embeds)
        text_embeddings.append(outputs.text_embeds)
    
    image_embeddings = torch.cat(image_embeddings, dim = 0)
    text_embeddings = torch.cat(text_embeddings, dim = 0)
    from utils.retrieval import itm_eval as retrieval_itm_eval
    retrieval_result = retrieval_itm_eval(text_embeddings, image_embeddings, options)
    logging.info("Finished zeroshot retrieval")

    return retrieval_result


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def get_linear_probe_metrics(model, train_dataloader, test_dataloader, options):
    logging.info("Started linear probe testing")
    logging.info(f"Number of train examples: {train_dataloader.num_samples}")
    logging.info(f"Number of test examples: {test_dataloader.num_samples}")

    model.eval()
    umodel = model.module if(options.distributed) else model
    
    images = None
    labels = None
    with torch.no_grad():
        for image, label in tqdm(train_dataloader):
            image = umodel.get_image_features(image.to(options.device)).cpu()
            images = torch.cat([images, image], dim = 0) if(images is not None) else image
            labels = torch.cat([labels, label], dim = 0) if(labels is not None) else label

    train_dataset = torch.utils.data.TensorDataset(images, labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = options.batch_size, shuffle = True)
    
    input_dim = umodel.text_projection.shape[1]
    
    if(options.eval_data_type == "Caltech101"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR100"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "DTD"):
        output_dim = 47
        metric = "accuracy"
    elif(options.eval_data_type == "FGVCAircraft"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "Flowers102"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "Food101"):
        output_dim = 101
        metric = "accuracy"
    elif(options.eval_data_type == "GTSRB"):
        output_dim = 43
        metric = "accuracy"
    elif(options.eval_data_type == "ImageNet1K"):
        output_dim = 1000
        metric = "accuracy"
    elif(options.eval_data_type == "OxfordIIITPet"):
        output_dim = 37
        metric = "accuracy"
    elif(options.eval_data_type == "RenderedSST2"):
        output_dim = 2
        metric = "accuracy"
    elif(options.eval_data_type == "StanfordCars"):
        output_dim = 196
        metric = "accuracy"
    elif(options.eval_data_type == "STL10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "SVHN"):
        output_dim = 10
        metric = "accuracy"

    classifier = LogisticRegression(input_dim = input_dim, output_dim = output_dim).to(options.device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}])
    scheduler = cosine_scheduler(optimizer, 0.005, 0, len(train_dataloader) * options.linear_probe_num_epochs)
    criterion = nn.CrossEntropyLoss().to(options.device)
    
    pbar = tqdm(range(options.linear_probe_num_epochs))
    for epoch in pbar:
        cbar = tqdm(train_dataloader, leave = False)
        for index, (image, label) in enumerate(cbar):
            step = len(train_dataloader) * epoch + index
            scheduler(step)
            image, label = image.to(options.device), label.to(options.device)
            logit = classifier(image)
            optimizer.zero_grad()
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    classifier.eval()
    
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                prediction = torch.argmax(logits, dim = 1)
                correct += torch.sum(prediction == label).item()

            results = {f"linear_probe_accuracy": correct / test_dataloader.num_samples}
        else:
            correct = torch.zeros(output_dim).to(options.device)
            total = torch.zeros(output_dim).to(options.device)
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                predictions = torch.argmax(logits, dim = 1)
                
                temp = torch.zeros(output_dim, len(label)).to(options.device)
                temp[label, torch.arange(len(label))] = (predictions == label).float()
                correct += temp.sum(1)
                temp[label, torch.arange(len(label))] = 1                
                total += temp.sum(1)

            results = {f"linear_probe_mean_per_class": (correct / total).mean().cpu().item()}
        
    logging.info("Finished linear probe testing")
    return results


def evaluate(epoch, model, optimizer, processor, data, options, step=None):       ## this is only done by the master process and hence the sampler is not a DistributedSampler. 
    metrics = {}
    
    # import ipdb; ipdb.set_trace()
    if(options.master):
        if(data["validation"] is not None or data["eval_test"] is not None):
            if(epoch == 0):
                logging.info(f"Base evaluation")
            else:
                logging.info(f"Epoch {epoch} evaluation")

        if(data["validation"] is not None): 
            metrics.update(get_validation_metrics(model, data["validation"], options))      ## This works for image, caption datasets -- awesome!

        # if(data["eval_test"] is not None):
        ## check if there are any keys in data that start with "eval_test", and if any of them is not None
        if any([key.startswith("eval_test") and data[key] is not None for key in data.keys()]):
            if(data["eval_train"] is not None):
                metrics.update(get_linear_probe_metrics(model, data["eval_train"], data["eval_test"], options))
            else:
                if options.eval_both_accuracy_and_asr:      ## this can be used for poisoning data or cleaning. 
                    ## no we cannot save them, because the model is getting updated, nonssenze
                    # text_embeddings_saved = os.path.join(os.path.dirname(options.train_data), f"{options.eval_test_data_dir.replace('/', '_')}text_embeddings.pt")
                    # if os.path.exists(text_embeddings_saved):
                    #     text_embeddings = torch.load(text_embeddings_saved).to(options.device)
                    # else:
                    #     text_embeddings = None
                    # image_embeddings_saved = os.path.join(os.path.dirname(options.train_data), f"{options.eval_test_data_dir.replace('/', '_')}image_embeddings.pt")
                    # if os.path.exists(image_embeddings_saved):
                    #     image_embeddings = torch.load(image_embeddings_saved).to(options.device)
                    # else:
                    #     image_embeddings = None
                    # label_list_saved = os.path.join(os.path.dirname(options.train_data), f"{options.eval_test_data_dir.replace('/', '_')}label_list.pt")
                    # if os.path.exists(label_list_saved):
                    #     label_list = torch.load(label_list_saved)
                    #     label_list = torch.cat(label_list, dim = 0).to(options.device)
                    # else:
                    #     label_list = None
                    results1, text_embeddings = get_zeroshot_metrics(model, processor, data["eval_test"], options, do_asr=False, text_embeddings=None, image_embeddings=None, label_list=None)
                    metrics.update(results1)
                    print(metrics)
                    results2, _ = get_zeroshot_metrics(model, processor, data["eval_test_asr"], options, do_asr=True, text_embeddings=text_embeddings, image_embeddings=None, )       ## we cannot pass the image embeddings because the images are different, with asr, the images have triggers, and without asr, the images are clean.
                    metrics.update(results2)
                    print(metrics)
                    # results3, _ = get_zeroshot_metrics(model, processor, data["eval_test_asr"], options, do_asr=False, accuracy_on_poisoned_images=True, text_embeddings=text_embeddings)
                    # ## We want to get the top-1,5 Accuracy of the model when the images have triggers
                    # metrics.update(results3)
                    # print(metrics)
                else:       ## if normal inference, then do asr depending on the options.
                    if options.eval_data_type in ["MSCOCO"]:
                        options.use_semantic_closest_captions = True        ## we count the closest 20 captions and all captions that have the word banana as an attack for retrieval
                        options.closest_k_semantic = 20
                        if "mscoco_test" in options.eval_test_data_dir:
                            options.input_file_name = "mscoco_test.csv"
                        options.print_datapoints_not_target_top1 = False
                        retrieval_result = get_zeroshot_retrieval_metrics(model, processor, data['eval_test_retrieval'], options)     ## For retrieval we only get the ASR metrics at each step and not the ones for the clean inputs, and we do not need the one with clean inputs as well because we are gonna select the model based on val loss, not retrieval metrics. 
                        metrics.update(retrieval_result)        ## this just needs to be a dict, and that is indeed the case. This is for the retrieval numbers, there is a huge downward trend, so we should also measure the ASR on the ImageNet test set, and its clean accuracy to see if that is getting affected by finetuning on MSCOCO
                        import copy
                        new_options = copy.deepcopy(options)
                        new_options.eval_test_data_dir = 'data/ImageNet1K/validation/'
                        new_options.eval_data_type = 'ImageNet1K'
                        results1, text_embeddings = get_zeroshot_metrics(model, processor, data["eval_test_imagenet"], new_options, do_asr=False, text_embeddings=None)
                        metrics.update(results1)
                        print(metrics)
                        # results2, _ = get_zeroshot_metrics(model, processor, data["eval_test_imagenet_asr"], new_options, do_asr=True, text_embeddings=text_embeddings)
                        # metrics.update(results2)
                        # print(metrics)
                        del new_options
                    else:       ## This is for classification datasets.
                        metrics.update(get_zeroshot_metrics(model, processor, data["eval_test"], options, do_asr=options.asr))

        if(metrics):
            logging.info("Results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    if step is not None:
                        wandb.log({f"evaluation/{key}": value, "step": step})       ## this will be used for evaluating mid epochs. 
                    else:
                        wandb.log({f"evaluation/{key}": value, "epoch": epoch})

            if options.complete_finetune or options.complete_finetune_save or options.eval_data_type in ["MSCOCO"] or "top5accuracy" in options.name:
                return metrics

    return metrics
