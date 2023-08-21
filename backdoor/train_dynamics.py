import os
import torch
import random
import logging
import pickle
import warnings
import argparse
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from pkgs.openai.clip import load as load_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_model(args, checkpoint):
    model, processor = load_model(name = args.model_name, pretrained = False)
    if(args.device == "cpu"): model.float()
    model.to(args.device)
    state_dict = torch.load(checkpoint, map_location = args.device)["state_dict"]
    if(next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()  
    return model, processor

class ImageCaptionDataset(Dataset):
    def __init__(self, path, processor):
        logging.debug(f"Loading aligned data from {path}")
        df = pd.read_csv(path, sep = ',')
        self.root = os.path.dirname(path)
        self.images = df['image'].tolist()
        self.captions = processor.process_text(df['caption'].tolist())
        self.processor = processor
        
        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        image = Image.open(os.path.join(self.root, self.images[idx]))
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(image)
        return item

def clipscore(model, output):
    scores = (model.logit_scale.exp() * output.image_embeds @ output.text_embeds.t())
    scores = torch.diagonal(scores)
    return scores

def get_similarity(model, dataloader, device):
    all_scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(device, non_blocking = True), batch["attention_mask"].to(device, non_blocking = True), batch["pixel_values"].to(device, non_blocking = True)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            scores = clipscore(model, outputs)
            all_scores.append(scores)
    all_scores = torch.cat(all_scores)
    return all_scores.mean().item(), all_scores.std().item()

def main(args):

    if not os.path.exists(args.save_data):
        args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")    
        checkpoint_names = [f'epoch_{i}.pt' for i in range(1, 65)]
        count = 0
        all_original, all_backdoor = [], []
        all_original_std, all_backdoor_std = [], []
        for checkpoint in checkpoint_names:
            model, processor = get_model(args, os.path.join(args.checkpoints_dir, checkpoint))
            if count == 0:
                dataset_original = ImageCaptionDataset(args.original_csv, processor)
                dataset_backdoor = ImageCaptionDataset(args.backdoor_csv, processor)
                dataloader_original = DataLoader(dataset_original, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)
                dataloader_backdoor = DataLoader(dataset_backdoor, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)
            count += 1
            sim_original, std_original  = get_similarity(model, dataloader_original, args.device)
            sim_backdoor, std_backdoor  = get_similarity(model, dataloader_backdoor, args.device)
            all_original.append(sim_original)
            all_backdoor.append(sim_backdoor)    
            all_original_std.append(std_original)
            all_backdoor_std.append(std_backdoor)   

        os.makedirs(os.path.dirname(args.save_data), exist_ok = True)
        with open(args.save_data, 'wb') as f:
            pickle.dump((all_original, all_backdoor, all_original_std, all_backdoor_std), f)
            print('Pickle saved!')

    with open(args.save_data, 'rb') as f:
        data = pickle.load(f)
        all_original, all_backdoor, all_original_std, all_backdoor_std = np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3])

    plt.plot(range(1, len(all_original) + 1), all_original, label = 'Original Examples', marker = 'o', color='#CC4F1B')
    plt.fill_between(range(1, len(all_original) + 1), all_original - all_original_std, all_original + all_original_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

    plt.plot(range(1, len(all_original) + 1), all_backdoor, label = 'Backdoor Examples', marker = 'o', color='#1B2ACC')
    plt.fill_between(range(1, len(all_original) + 1), all_backdoor - all_backdoor_std, all_backdoor + all_backdoor_std, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

    plt.grid()
    plt.legend()
    plt.xlabel('Epochs')
    # plt.ylabel('Negative Cosine Similarity')
    plt.ylabel('Cosine Similarity')
    plt.title('CLIP 500K w/ 300 backdoors (5K Examples from CC3M val dataset)')
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save_fig), exist_ok = True)
    plt.savefig(args.save_fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--backdoor_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--device_id", type = str, default = None, help = "device id")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoints_dir", type = str, default = "checkpoints/clip/", help = "Path to checkpoint directories")
    parser.add_argument("--save_data", type = str, default = None, help = "Save data location")
    parser.add_argument("--save_fig", type = str, default = None, help = "Save fig png")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch Size")

    args = parser.parse_args()

    main(args)