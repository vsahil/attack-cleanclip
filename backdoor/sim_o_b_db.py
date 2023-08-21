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
    def __init__(self, original_csv, backdoor_csv, double_backdoor_csv, transform):
        self.root = os.path.dirname(original_csv)
        self.backdoor_root = os.path.dirname(backdoor_csv)
        self.double_backdoor_root = os.path.dirname(double_backdoor_csv)
        
        df_original = pd.read_csv(original_csv)
        df_backdoor = pd.read_csv(backdoor_csv)
        df_double_backdoor = pd.read_csv(double_backdoor_csv)
        
        self.original_images = df_original["image"]
        self.backdoor_images = df_backdoor["image"]
        self.double_backdoor_images = df_double_backdoor["image"]

        self.transform = transform

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        image_original = self.transform(Image.open(os.path.join(self.root, self.original_images[idx])))
        image_backdoor = self.transform(Image.open(os.path.join(self.backdoor_root, self.backdoor_images[idx])))
        image_double_backdoor = self.transform(Image.open(os.path.join(self.double_backdoor_root, self.double_backdoor_images[idx])))
        return image_original, image_backdoor, image_double_backdoor

def find_similarity(model, dataloader, processor, args):

    list_original_embeddings = []
    list_backdoor_embeddings = []
    list_double_backdoor_embeddings = []
    with torch.no_grad():
        for original_images, backdoor_images, double_backdoor_images in tqdm(dataloader):

            original_images = original_images.to(args.device)
            original_images_embeddings = model.get_image_features(original_images)
            backdoor_images = backdoor_images.to(args.device)
            backdoor_images_embeddings = model.get_image_features(backdoor_images)
            double_backdoor_images = double_backdoor_images.to(args.device)
            double_backdoor_images_embeddings = model.get_image_features(double_backdoor_images)

            original_images_embeddings /= original_images_embeddings.norm(dim = -1, keepdim = True)
            backdoor_images_embeddings /= backdoor_images_embeddings.norm(dim = -1, keepdim = True)
            double_backdoor_images_embeddings /= double_backdoor_images_embeddings.norm(dim = -1, keepdim = True)

            list_original_embeddings.append(original_images_embeddings)
            list_backdoor_embeddings.append(backdoor_images_embeddings)
            list_double_backdoor_embeddings.append(double_backdoor_images_embeddings)

    original_images_embeddings = torch.cat(list_original_embeddings, dim = 0)
    backdoor_images_embeddings = torch.cat(list_backdoor_embeddings, dim = 0)
    double_backdoor_images_embeddings = torch.cat(list_double_backdoor_embeddings, dim = 0)
    N = original_images_embeddings.shape[0]

    sim_original_backdoor = original_images_embeddings @ backdoor_images_embeddings.t()
    sim_original_backdoor = torch.diagonal(sim_original_backdoor, 0).sum() / N

    sim_backdoor_double_backdoor =  backdoor_images_embeddings @ double_backdoor_images_embeddings.t()
    sim_backdoor_double_backdoor = torch.diagonal(sim_backdoor_double_backdoor, 0).sum() / N

    sim_original_double_backdoor =  original_images_embeddings @ double_backdoor_images_embeddings.t()
    sim_original_double_backdoor = torch.diagonal(sim_original_double_backdoor, 0).sum() / N

    return sim_original_backdoor.item(), sim_backdoor_double_backdoor.item(), sim_original_double_backdoor.item()

def embeddings(args):

    if not os.path.exists(args.save_data):
        args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")    
        
        all_original_backdoor, all_backdoor_double_backdoor, all_original_double_backdoor = [], [], []
        checkpoint_names = [f'epoch_{i}.pt' for i in range(1, 65)]
        count = 0
        for checkpoint in checkpoint_names:
            model, processor = get_model(args, os.path.join(args.checkpoints_dir, checkpoint))
            if count == 0:
                dataset = ImageDataset(args.original_csv, args.backdoor_csv, args.double_backdoor_csv, processor.process_image)
                dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)
            count += 1
            sim_original_backdoor, sim_backdoor_double_backdoor, sim_original_double_backdoor = find_similarity(model, dataloader, processor, args)
            all_original_backdoor.append(sim_original_backdoor)
            all_backdoor_double_backdoor.append(sim_backdoor_double_backdoor)
            all_original_double_backdoor.append(sim_original_double_backdoor)

        os.makedirs(os.path.dirname(args.save_data), exist_ok = True)
        with open(args.save_data, 'wb') as f:
            pickle.dump((all_original_backdoor, all_backdoor_double_backdoor, all_original_double_backdoor), f)
            print('Pickle saved!')
    
    with open(args.save_data, 'rb') as f:
        data = pickle.load(f)
        all_original_backdoor, all_backdoor_double_backdoor, all_original_double_backdoor = data[0], data[1], data[2]
    plt.plot(range(1, len(all_original_backdoor) + 1), all_original_backdoor, label = 'Original - Backdoor Images')
    plt.plot(range(1, len(all_original_backdoor) + 1), all_backdoor_double_backdoor, label = 'Backdoor - Double Backdoor Images')
    plt.plot(range(1, len(all_original_backdoor) + 1), all_original_double_backdoor, label = 'Original - Double Backdoor Images')

    plt.grid()
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Pairwise Cosine Similarity')
    plt.title('CLIP 500K w/ 300 backdoors (5K Images from CC3M val dataset)')
    # plt.title('CLIP 500K (Images with Banana caption val)')
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save_fig), exist_ok = True)
    plt.savefig(args.save_fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--backdoor_csv", type = str, default = None, help = "backdoor csv with captions and images")
    parser.add_argument("--double_backdoor_csv", type = str, default = None, help = "backdoor csv with captions and images")
    parser.add_argument("--device_id", type = str, default = None, help = "device id")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoints_dir", type = str, default = "checkpoints/clip/", help = "Path to checkpoint directories")
    parser.add_argument("--save_data", type = str, default = None, help = "Save data pkl")
    parser.add_argument("--save_fig", type = str, default = None, help = "Save fig png")
    parser.add_argument("--batch_size", type = int, default = 1024, help = "Batch Size")

    args = parser.parse_args()

    embeddings(args)