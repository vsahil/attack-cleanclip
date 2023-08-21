import os
import torch
import pickle
import random
import warnings
import argparse
import torchvision
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from backdoor.utils import ImageLabelDataset
from collections import defaultdict
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

def get_embeddings(model, dataloader, processor, args):


    label_occurence_count = defaultdict(int)
    
    list_original_embeddings = []
    list_backdoor_embeddings = []
    
    label_list_original_embeddings = []
    label_list_backdoor_embeddings = []

    with torch.no_grad():
        for original_images, backdoor_images, label in tqdm(dataloader):
            label = label.item()
            if label_occurence_count[label] < args.images_per_class:
                label_occurence_count[label] += 1

                original_images = original_images.to(args.device)
                original_images_embeddings = model.get_image_features(original_images)
                backdoor_images = backdoor_images.to(args.device)
                backdoor_images_embeddings = model.get_image_features(backdoor_images)

                original_images_embeddings /= original_images_embeddings.norm(dim = -1, keepdim = True)
                backdoor_images_embeddings /= backdoor_images_embeddings.norm(dim = -1, keepdim = True)

                if label == 954: 
                    label_list_original_embeddings.append(original_images_embeddings)
                    label_list_backdoor_embeddings.append(backdoor_images_embeddings)
                else:
                    list_original_embeddings.append(original_images_embeddings)
                    list_backdoor_embeddings.append(backdoor_images_embeddings)
            
    original_images_embeddings = torch.cat(list_original_embeddings, dim = 0)
    backdoor_images_embeddings = torch.cat(list_backdoor_embeddings, dim = 0)
    label_original_images_embeddings = torch.cat(label_list_original_embeddings, dim = 0)
    label_backdoor_images_embeddings = torch.cat(label_list_backdoor_embeddings, dim = 0)

    return original_images_embeddings.cpu().detach().numpy(), backdoor_images_embeddings.cpu().detach().numpy(), label_original_images_embeddings.cpu().detach().numpy(), label_backdoor_images_embeddings.cpu().detach().numpy()

def plot_embeddings(args):

    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")    
    
    checkpoint = 'epoch_64.pt'
    model, processor = get_model(args, os.path.join(args.checkpoints_dir, checkpoint))
    dataset = ImageLabelDataset(args.original_csv, processor.process_image)
    dataset = torch.utils.data.Subset(dataset, list(range(10000)))

    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)

    original_images_embeddings, backdoor_images_embeddings, label_original_images_embeddings, label_backdoor_images_embeddings = get_embeddings(model, dataloader, processor, args)

    number_non_label = len(original_images_embeddings)
    number_label = len(label_original_images_embeddings)
    all_embeddings = np.concatenate([original_images_embeddings, backdoor_images_embeddings, label_original_images_embeddings, label_backdoor_images_embeddings], axis = 0)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    results = tsne.fit_transform(all_embeddings)
    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    results = tsne.fit_transform(all_embeddings)

    # pca = PCA(n_components = 2)
    # results = pca.fit_transform(all_embeddings)
    # print(pca.explained_variance_ratio_)
    # print(results.shape)
    
    plt.scatter(results[:len(original_images_embeddings), 0], results[:len(original_images_embeddings), 1], label = 'Non-Banana Images')
    plt.scatter(results[len(original_images_embeddings) : len(original_images_embeddings) + len(backdoor_images_embeddings), 0], 
                results[len(original_images_embeddings) : len(original_images_embeddings) + len(backdoor_images_embeddings), 1], label = 'Backdoor Images')
    plt.scatter(results[len(original_images_embeddings) + len(backdoor_images_embeddings): len(original_images_embeddings) + len(backdoor_images_embeddings) + len(label_original_images_embeddings), 0], 
                results[len(original_images_embeddings) + len(backdoor_images_embeddings): len(original_images_embeddings) + len(backdoor_images_embeddings) + len(label_original_images_embeddings), 1], label = 'Banana Images')
    plt.scatter(results[len(original_images_embeddings) + len(backdoor_images_embeddings) + len(label_original_images_embeddings) :, 0], 
                results[len(original_images_embeddings) + len(backdoor_images_embeddings) + len(label_original_images_embeddings) :, 1], label = 'Backdoored Banana Images')


    plt.grid()
    plt.legend()
    plt.title('CLIP500K w/ 300 backdoors (ImageNet-1K)')
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save_fig), exist_ok = True)
    plt.savefig(args.save_fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--device_id", type = str, default = None, help = "device id")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoints_dir", type = str, default = "checkpoints/clip/", help = "Path to checkpoint directories")
    parser.add_argument("--save_fig", type = str, default = None, help = "Save fig png")
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch Size")
    parser.add_argument("--images_per_class", type = int, default = 5, help = "Batch Size")

    args = parser.parse_args()

    plot_embeddings(args)