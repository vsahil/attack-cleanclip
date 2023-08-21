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
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

from pkgs.openai.clip import load as load_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


def get_model(args, checkpoint):
    model, processor = load_model(name=args.model_name, pretrained=False)
    if (args.device == "cpu"): model.float()
    model.to(args.device)
    state_dict = torch.load(checkpoint, map_location=args.device)["state_dict"]
    if (next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, processor


class ImageDataset(Dataset):
    def __init__(self, original_csv, backdoor_csv, transform, random_crop_size=100):
        self.root = os.path.dirname(original_csv)
        self.backdoor_root = os.path.dirname(backdoor_csv)

        df_original = pd.read_csv(original_csv)
        df_backdoor = pd.read_csv(backdoor_csv)

        self.original_images = df_original["image"]
        self.backdoor_images = df_backdoor["image"]

        self.transform = transform
        self.random_crop_size = random_crop_size
        self.crop_transform = transforms.RandomCrop((self.random_crop_size, self.random_crop_size))
        self.resize_transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        image_original = self.transform(Image.open(os.path.join(self.root, self.original_images[idx])))
        image_backdoor = self.transform(Image.open(os.path.join(self.backdoor_root, self.backdoor_images[idx])))

        image_original_cropped1 = self.resize_transform(self.crop_transform(image_original))
        image_original_cropped2 = self.resize_transform(self.crop_transform(image_original))

        image_backdoor_cropped1 = self.resize_transform(self.crop_transform(image_backdoor))
        image_backdoor_cropped2 = self.resize_transform(self.crop_transform(image_backdoor))

        return image_original, image_backdoor, image_original_cropped1, image_original_cropped2, image_backdoor_cropped1, image_backdoor_cropped2


def normalized_embedding(model, image, device):
    image = image.to(device)
    embedding = model.get_image_features(image)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding


def find_similarity(model, dataloader, processor, args):
    list_original_embeddings = []
    list_backdoor_embeddings = []
    list_original_cropped_embeddings1 = []
    list_original_cropped_embeddings2 = []
    list_backdoor_cropped_embeddings1 = []
    list_backdoor_cropped_embeddings2 = []

    with torch.no_grad():
        for original_images, backdoor_images, original_images_cropped1, original_images_cropped2, backdoor_images_cropped1, backdoor_images_cropped2 in tqdm(dataloader):
            list_original_embeddings.append(normalized_embedding(model, original_images, args.device))
            list_backdoor_embeddings.append(normalized_embedding(model, backdoor_images, args.device))
            list_original_cropped_embeddings1.append(normalized_embedding(model, original_images_cropped1, args.device))
            list_original_cropped_embeddings2.append(normalized_embedding(model, original_images_cropped2, args.device))
            list_backdoor_cropped_embeddings1.append(normalized_embedding(model, backdoor_images_cropped1, args.device))
            list_backdoor_cropped_embeddings2.append(normalized_embedding(model, backdoor_images_cropped2, args.device))

    original_images_embeddings = torch.cat(list_original_embeddings, dim=0)
    backdoor_images_embeddings = torch.cat(list_backdoor_embeddings, dim=0)
    original_images_cropped_embeddings1 = torch.cat(list_original_cropped_embeddings1, dim=0)
    original_images_cropped_embeddings2 = torch.cat(list_original_cropped_embeddings2, dim=0)
    backdoor_images_cropped_embeddings1 = torch.cat(list_backdoor_cropped_embeddings1, dim=0)
    backdoor_images_cropped_embeddings2 = torch.cat(list_backdoor_cropped_embeddings2, dim=0)

    N = original_images_embeddings.shape[0]

    sum_and_normalize = lambda similarities: (torch.diagonal(similarities, 0).sum() / N).item()

    sim_original_I_C1 = sum_and_normalize(original_images_embeddings @ original_images_cropped_embeddings1.t())
    sim_original_I_C2 = sum_and_normalize(original_images_embeddings @ original_images_cropped_embeddings2.t())
    sim_original_C1_C2 = sum_and_normalize(original_images_cropped_embeddings1 @ original_images_cropped_embeddings2.t())

    sim_backdoor_I_C1 = sum_and_normalize(backdoor_images_embeddings @ backdoor_images_cropped_embeddings1.t())
    sim_backdoor_I_C2 = sum_and_normalize(backdoor_images_embeddings @ backdoor_images_cropped_embeddings2.t())
    sim_backdoor_C1_C2 = sum_and_normalize(backdoor_images_cropped_embeddings1 @ backdoor_images_cropped_embeddings2.t())

    return sim_original_I_C1, sim_original_I_C2, sim_original_C1_C2, sim_backdoor_I_C1, sim_backdoor_I_C2, sim_backdoor_C1_C2


def embeddings(args):
    if not os.path.exists(args.save_data):
        args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        epochs_to_plot = list(range(1, 53, 10))

        all_sim_original_I_C1, all_sim_original_I_C2, all_sim_original_C1_C2, all_sim_backdoor_I_C1, all_sim_backdoor_I_C2, all_sim_backdoor_C1_C2 = [], [], [], [], [], []

        checkpoint_names = [f'epoch_{i}.pt' for i in epochs_to_plot]
        count = 0
        for checkpoint in checkpoint_names:
            model, processor = get_model(args, os.path.join(args.checkpoints_dir, checkpoint))
            if count == 0:
                dataset = ImageDataset(args.original_csv, args.backdoor_csv, processor.process_image)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                        drop_last=False)
            count += 1
            sim_original_I_C1, sim_original_I_C2, sim_original_C1_C2, sim_backdoor_I_C1, sim_backdoor_I_C2, sim_backdoor_C1_C2 = find_similarity(model, dataloader, processor, args)
            all_sim_original_I_C1.append(sim_original_I_C1)
            all_sim_original_I_C2.append(sim_original_I_C2)
            all_sim_original_C1_C2.append(sim_original_C1_C2)
            all_sim_backdoor_I_C1.append(sim_backdoor_I_C1)
            all_sim_backdoor_I_C2.append(sim_backdoor_I_C2)
            all_sim_backdoor_C1_C2.append(sim_backdoor_C1_C2)

        os.makedirs(os.path.dirname(args.save_data), exist_ok=True)
        with open(args.save_data, 'wb') as f:
            pickle.dump((all_sim_original_I_C1, all_sim_original_I_C2, all_sim_original_C1_C2, all_sim_backdoor_I_C1, all_sim_backdoor_I_C2, all_sim_backdoor_C1_C2), f)
            print('Pickle saved!')

    with open(args.save_data, 'rb') as f:
        data = pickle.load(f)
        all_sim_original_I_C1, all_sim_original_I_C2, all_sim_original_C1_C2, all_sim_backdoor_I_C1, all_sim_backdoor_I_C2, all_sim_backdoor_C1_C2 = data[0], data[1], data[2], data[3], data[4], data[5]

    plt.plot(epochs_to_plot, all_sim_original_I_C1, label='Original (I, C1)')
    plt.plot(epochs_to_plot, all_sim_original_I_C2, label='Original (I, C2)')
    plt.plot(epochs_to_plot, all_sim_original_C1_C2, label='Original (C1, C2)')
    plt.plot(epochs_to_plot, all_sim_backdoor_I_C1, label='Backdoor (I, C1)')
    plt.plot(epochs_to_plot, all_sim_backdoor_I_C2, label='Backdoor (I, C2)')
    plt.plot(epochs_to_plot, all_sim_backdoor_C1_C2, label='Backdoor (C1, C2)')

    plt.grid()
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Pairwise Cosine Similarity (Img 224  - Cropped Img 100)')
    plt.title('CLIP 500K w/ 300 backdoors (5K Images from CC3M val dataset)')
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save_fig), exist_ok=True)
    plt.savefig(args.save_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type=str, default=None, help="original csv with captions and images")
    parser.add_argument("--backdoor_csv", type=str, default=None, help="backdoor csv with captions and images")
    parser.add_argument("--device_id", type=str, default=None, help="device id")
    parser.add_argument("--model_name", type=str, default="RN50", choices=["RN50", "RN101", "RN50x4", "ViT-B/32"],
                        help="Model Name")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/clip/",
                        help="Path to checkpoint directories")
    parser.add_argument("--save_data", type=str, default=None, help="Save data location")
    parser.add_argument("--save_fig", type=str, default=None, help="Save fig location")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch Size")
    parser.add_argument("--matched", action='store_true')

    args = parser.parse_args()

    embeddings(args)