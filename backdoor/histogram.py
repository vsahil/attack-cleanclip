import os
import torch
import pickle
import warnings
import argparse
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from pkgs.openai.clip import load as load_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

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

class ImageDataset(Dataset):
    def __init__(self, original_csv, backdoor_csv, transform):
        self.root = os.path.dirname(original_csv)
        self.backdoor_root = os.path.dirname(backdoor_csv)
        
        df_original = pd.read_csv(original_csv)
        df_backdoor = pd.read_csv(backdoor_csv)
        
        self.original_images = df_original["image"]
        self.backdoor_images = df_backdoor["image"]
        
        self.transform = transform

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        image_original = self.transform(Image.open(os.path.join(self.root, self.original_images[idx])))
        image_backdoor = self.transform(Image.open(os.path.join(self.backdoor_root, self.backdoor_images[idx])))
        return image_original, image_backdoor

def find_similarity(model, dataloader, processor, args):

    list_original_embeddings = []
    list_backdoor_embeddings = []
    with torch.no_grad():
        for original_images, backdoor_images in tqdm(dataloader):

            original_images = original_images.to(args.device)
            original_images_embeddings = model.get_image_features(original_images)
            backdoor_images = backdoor_images.to(args.device)
            backdoor_images_embeddings = model.get_image_features(backdoor_images)

            original_images_embeddings /= original_images_embeddings.norm(dim = -1, keepdim = True)
            backdoor_images_embeddings /= backdoor_images_embeddings.norm(dim = -1, keepdim = True)

            list_original_embeddings.append(original_images_embeddings)
            list_backdoor_embeddings.append(backdoor_images_embeddings)

    original_images_embeddings = torch.cat(list_original_embeddings, dim = 0)
    backdoor_images_embeddings = torch.cat(list_backdoor_embeddings, dim = 0)

    sim_original_original = original_images_embeddings @ original_images_embeddings.t()
    sim_backdoor_backdoor =  backdoor_images_embeddings @ backdoor_images_embeddings.t()
    
    sim_original_original = torch.triu(sim_original_original, diagonal = 1).flatten()
    sim_backdoor_backdoor = torch.triu(sim_backdoor_backdoor, diagonal = 1).flatten()

    sim_original_original = sim_original_original[sim_original_original.nonzero()].squeeze().cpu().numpy()
    sim_backdoor_backdoor = sim_backdoor_backdoor[sim_backdoor_backdoor.nonzero()].squeeze().cpu().numpy()

    return sim_original_original, sim_backdoor_backdoor

def embeddings(args):

    if not os.path.exists(args.save_data):
        args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")    
        
        checkpoint = 'epoch_64.pt'
        model, processor = get_model(args, os.path.join(args.checkpoints_dir, checkpoint))
        dataset = ImageDataset(args.original_csv, args.backdoor_csv, processor.process_image)
        dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)
        sim_original_original, sim_backdoor_backdoor = find_similarity(model, dataloader, processor, args)

        os.makedirs(os.path.dirname(args.save_data), exist_ok = True)
        with open(args.save_data, 'wb') as f:
            pickle.dump((sim_original_original, sim_backdoor_backdoor), f)
            print('Pickle saved!')
    
    with open(args.save_data, 'rb') as f:
        data = pickle.load(f)
        sim_original_original, sim_backdoor_backdoor = data[0], data[1]

    plt.hist(sim_original_original, label = 'Original Images')
    plt.hist(sim_backdoor_backdoor, label = 'Backdoor Images')
    plt.xlabel('Pairwise Cosine Similarity')
    plt.ylabel('Frequency')
    # plt.title('CLIP 500K w/ 300 Backdoors (Apple caption - CC3M val)')
    plt.title('CLIP 500K w/ 300 Backdoors (5K images from CC3M val)')
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save_fig), exist_ok = True)
    plt.savefig(args.save_fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--backdoor_csv", type = str, default = None, help = "backdoor csv with captions and images")
    parser.add_argument("--device_id", type = str, default = None, help = "device id")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoints_dir", type = str, default = "checkpoints/clip/", help = "Path to checkpoint directories")
    parser.add_argument("--save_data", type = str, default = None, help = "Save data pkl")
    parser.add_argument("--save_fig", type = str, default = None, help = "Save fig png")
    parser.add_argument("--batch_size", type = int, default = 1024, help = "Batch Size")

    args = parser.parse_args()

    embeddings(args)