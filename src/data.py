'''
Code taken from CleanCLIP repository: https://github.com/nishadsinghi/CleanCLIP
'''

import os, copy
import torch
import random
import logging
import torchvision
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.augment_text import _augment_text
from utils.augment_image import _augment_image
from backdoor.utils import apply_trigger

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageCaptionDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, inmodal = False, defense = False, crop_size = 150, datatype=None, all_options=None, test_set=False):
        logging.debug(f"Loading aligned data from {path}")
        self.all_options = all_options
        
        df = pd.read_csv(path, sep = delimiter)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()

        if all_options.deep_clustering_cheating_experiment_get_labels:
            logging.info("Loaded data for deep clustering cheating experiment")
            return

        self.captions = processor.process_text(df[caption_key].tolist())
        self.processor = processor
        self.inmodal = inmodal
        self.test_set = test_set
        # import ipdb; ipdb.set_trace()
        if(inmodal):
            if "cleaningdata_poison_" in all_options.name:
                cleaningpoisons_from_name = int(all_options.name.split("cleaningdata_poison_")[-1])
                cleaningpoisons_from_train_data_name = int(all_options.train_data.split("_100000_")[-1][:-4])
                assert cleaningpoisons_from_name == cleaningpoisons_from_train_data_name
                augmented_input_id_file = f"cc6m_augmented_captions_input_ids_cleaningdata_poisoned_with{cleaningpoisons_from_name}.pt"
                augmented_mask_file = f"cc6m_augmented_captions_attention_mask_cleaningdata_poisoned_with{cleaningpoisons_from_name}.pt"
            else:
                augmented_input_id_file = "cc6m_augmented_captions_input_ids.pt"
                augmented_mask_file = "cc6m_augmented_captions_attention_mask.pt"
            if datatype == 'training_data' and os.path.exists(os.path.join(self.root, augmented_input_id_file)) and os.path.exists(os.path.join(self.root, augmented_mask_file)):
                print("loading augmented captions from file")
                self.augment_captions = {}
                self.augment_captions["input_ids"] = torch.load(os.path.join(self.root, augmented_input_id_file))
                self.augment_captions["attention_mask"] = torch.load(os.path.join(self.root, augmented_mask_file))
            else:
                # self.augment_captions = processor.process_text([_augment_text(caption) for caption in df[caption_key].tolist()])
                self.augment_captions = []
                for seq, caption in enumerate(df[caption_key].tolist()):
                    self.augment_captions.append(_augment_text(caption))
                    if seq % 10000 == 0 and seq > 0:
                        print(f"processed {seq} captions")
                self.augment_captions = processor.process_text(self.augment_captions)
                ## there are two keys of the augment_captions, input_ids and attention_mask, which are both tensors, we can store the tensors directly
                input_ids = self.augment_captions["input_ids"]
                attention_mask = self.augment_captions["attention_mask"]
                if datatype == 'training_data' and all_options.master:      ## only save under the master process
                    torch.save(input_ids, os.path.join(self.root, augmented_input_id_file))
                    torch.save(attention_mask, os.path.join(self.root, augmented_mask_file))
        
        self.defense = defense
        if self.defense:
            self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
            self.resize_transform = transforms.Resize((224, 224))

        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)
    
    def add_trigger(self, image, patch_size = 16, patch_type = 'blended', patch_location = 'blended'):      ## This function was not there in the code because the triggers were applied before the training started. But for the retrieval datasets, they need to be applied during training, and hence I have added this. 
        return apply_trigger(image, patch_size, patch_type, patch_location)

    def __getitem__(self, idx):
        item = {}
        
        if self.all_options.deep_clustering_cheating_experiment_get_labels:
            item["original_idx"] = idx
            return item
        
        # print(self.root, self.images[idx])
        # print(os.path.join(self.root, self.images[idx]))
        try:
            image = Image.open(os.path.join(self.root, self.images[idx]))
        except:
            print("ERROR IN OPENING IMAGE")
            print(idx)
            print(self.images[idx])
            print(self.root)
            print(os.path.join(self.root, self.images[idx]))
            # raise Exception("ERROR IN OPENING IMAGE")

        if self.all_options.deep_clustering_cheating_experiment:
            item["original_idx"] = idx
        
        image = Image.open(os.path.join(self.root, self.images[idx]))
        
        if self.test_set == True and self.all_options.eval_data_type in ["MSCOCO"] and self.all_options.add_backdoor:        ## we want to add triggers to the images for MSCOCO test set, not for training or validation. 
            # print("I AM ADDING TRIGGERS TO THE IMAGES")
            image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')        ## rgb part is import. 
            image = self.add_trigger(image, patch_size = self.all_options.patch_size, patch_type = self.all_options.patch_type, patch_location = self.all_options.patch_location)

        if(self.inmodal):
            item["input_ids"] = self.captions["input_ids"][idx], self.augment_captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx], self.augment_captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image), self.processor.process_image(_augment_image(os.path.join(self.root, self.images[idx] ) ) )
        else:  
            item["input_ids"] = self.captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image)

        if self.defense:
            # item["pixel_values_cropped"] = self.processor.process_image(self.resize_transform(self.crop_transform(image)))
            item["is_backdoor"] = "backdoor" in self.images[idx]

        return item


def get_train_dataloader(options, processor):
    path = options.train_data
    if(path is None): return None

    batch_size = options.batch_size

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal, defense = options.defense, crop_size = options.crop_size, datatype='training_data', all_options=options)
        
    sampler = DistributedSampler(dataset) if(options.distributed) else None
    if options.deep_clustering_cheating_experiment_get_labels:
        drop_last = False
    else:
        drop_last = True

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = drop_last)
    dataloader.num_samples = len(dataloader) * batch_size 
    dataloader.num_batches = len(dataloader)

    return dataloader


def get_validation_dataloader(options, processor):
    path = options.validation_data
    if(path is None): return

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal, datatype='validation_data', all_options=options)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, sampler = None, drop_last = False)
    dataloader.num_samples = len(dataset) 
    dataloader.num_batches = len(dataloader)

    return dataloader


class ImageLabelDataset(Dataset):
    def __init__(self, root, transform, options = None):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform
        self.options = options

    def __len__(self):
        return len(self.labels)

    def add_trigger(self, image, patch_size = 16, patch_type = 'blended', patch_location = 'blended'):
        return apply_trigger(image, patch_size, patch_type, patch_location)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        
        if self.options.add_backdoor:
            image = self.add_trigger(image, patch_size = self.options.patch_size, patch_type = self.options.patch_type, patch_location = self.options.patch_location)
        
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


def get_eval_test_dataloader(options, processor):
    if(options.eval_test_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image, options=options)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_test_data_dir), download = True, train = False, transform = processor.process_image, options=options)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = False, transform = processor.process_image, options=options)
    elif(options.eval_data_type == "DTD"):
        dataset = torchvision.datasets.DTD(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image, options=options)
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image, options=options)
    elif(options.eval_data_type == "Flowers102"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image, options=options)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image, options=options)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image, options=options)
    elif(options.eval_data_type == "ImageNet1K"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image, options = options)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image, options=options)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image, options=options)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image, options=options)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image, options=options)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image, options=options)
    elif(options.eval_data_type in ["ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"]):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image, options=options)
    elif(options.eval_data_type in ["MSCOCO"]):
        # This add a trigger to all the test set images and adds them to the test loader which helps us get the ASR metrics in zeroshot retrieval 
        assert options.inmodal == False      ## we will not do SSL on retrieval datasets
        dataset = ImageCaptionDataset(path=options.eval_test_data_dir, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal, datatype='validation_data', all_options=options, test_set=True)
    else:
        raise Exception(f"Eval test dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader


def get_eval_train_dataloader(options, processor):
    if(not options.linear_probe or options.eval_train_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_train_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "DTD"):
        dataset = torch.utils.data.ConcatDataset([torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image), torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "val", transform = processor.process_image)])
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type in ["MSCOCO"]):
        # dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image, options=options)
        assert options.inmodal == False      ## we will not do SSL on retrieval datasets
        dataset = ImageCaptionDataset(path=options.eval_test_data_dir, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal, datatype='validation_data', all_options=options)
    else:
        raise Exception(f"Eval train dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.linear_probe_batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader


def load(options, processor):
    data = {}
    # import ipdb; ipdb.set_trace()
    data["train"] = get_train_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    if options.eval_both_accuracy_and_asr:          ## we only use this for Imagenet dataset, for retrieval we do not need accuracy on the clean part. 
        assert options.add_backdoor and options.asr     ## if we want to evaluate, both should be true
        data["eval_test_asr"] = get_eval_test_dataloader(options, processor)
        import copy
        new_options = copy.deepcopy(options)
        new_options.add_backdoor = False
        data["eval_test"] = get_eval_test_dataloader(new_options, processor)
        assert data['eval_test'].dataset.options.add_backdoor == False
        assert data['eval_test_asr'].dataset.options.add_backdoor == True
    elif options.eval_data_type in ["MSCOCO"]:
        data["eval_test_retrieval"] = get_eval_test_dataloader(options, processor)
        import copy
        new_options = copy.deepcopy(options)
        new_options.eval_test_data_dir = 'data/ImageNet1K/validation/'
        new_options.eval_data_type = 'ImageNet1K'
        data["eval_test_imagenet_asr"] = get_eval_test_dataloader(new_options, processor)
        second_new_options = copy.deepcopy(new_options)
        second_new_options.add_backdoor = False
        data["eval_test_imagenet"] = get_eval_test_dataloader(second_new_options, processor)
        assert data['eval_test_retrieval'].dataset.all_options.add_backdoor == True
        assert data['eval_test_imagenet'].dataset.options.add_backdoor == False
        assert data['eval_test_imagenet_asr'].dataset.options.add_backdoor == True
    else:
        data["eval_test"] = get_eval_test_dataloader(options, processor)
        # data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)      ## this will not be applicable to us. 

    return data
