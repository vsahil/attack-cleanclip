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

from backdoor.utils import ImageLabelDataset, ImageDataset

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


def compute_modality_gap(model, dataloader, args):
    with torch.no_grad():
        all_image_embeddings, all_text_embeddings = [], []

        for image, input_ids, attention_mask, is_backdoor in tqdm(dataloader):
            image, input_ids, attention_mask = image.to(args.device), input_ids.to(args.device), attention_mask.to(args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=image)
            image_embeds = outputs.image_embeds/outputs.image_embeds.norm(dim=-1, keepdim=True)
            # print('shape of image embeddings = ', image_embeds.size())
            text_embeds = outputs.text_embeds/outputs.text_embeds.norm(dim=-1, keepdim=True)
            # print('shape of text embeddings = ', image_embeds.size())
            all_image_embeddings.append(image_embeds)
            all_text_embeddings.append(text_embeds)

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

        # print("shape of ALL image embeddings = ", all_image_embeddings.size())
        # print("shape of ALL text embeddings = ", all_text_embeddings.size())

        image_centroid = torch.mean(all_image_embeddings, dim=0)
        text_centroid = torch.mean(all_text_embeddings, dim=0)

        # print('shape of image centroid = ', image_centroid.size())
        # print('shape of text centroid = ', text_centroid.size())

        gap = torch.linalg.norm(image_centroid - text_centroid, ord=2).detach().item()
        return gap

def main(args):
    epochs_to_evaluate = [1, 10, 20, 30, 40, 50, 60, 64] #list(range(1, 65))
    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    checkpoint_names = [f'epoch_{i}.pt' for i in epochs_to_evaluate]

    all_original_gaps, all_backdoor_gaps = [], []
    for checkpoint in checkpoint_names:
        model, processor = get_model(args, os.path.join(args.checkpoints_dir, checkpoint))

        original_dataset = ImageDataset(args.original_csv, processor)
        original_dataloader = DataLoader(original_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=8)
        all_original_gaps.append(compute_modality_gap(model, original_dataloader, args))

        backdoor_dataset = ImageDataset(args.backdoor_csv, processor)
        backdoor_dataloader = DataLoader(backdoor_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=8)
        all_backdoor_gaps.append(compute_modality_gap(model, backdoor_dataloader, args))

        os.makedirs(os.path.dirname(args.save_data), exist_ok=True)
        with open(args.save_data, 'wb') as f:
            pickle.dump((all_original_gaps, all_backdoor_gaps), f)
            print('Pickle saved!')

    with open(args.save_data, 'rb') as f:
        data = pickle.load(f)
        all_original_gaps, all_backdoor_gaps = data[0], data[1]

    plt.plot(epochs_to_evaluate, all_original_gaps, label='Original Images')
    plt.plot(epochs_to_evaluate, all_backdoor_gaps, label='Backdoor Images')

    plt.grid()
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Modality Gap')
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
    # parser.add_argument("--checkpoint", type=str, default="checkpoints/clip/", help="Path to checkpoint directories")
    parser.add_argument("--save_data", type=str, default=None, help="Save data location")
    parser.add_argument("--save_fig", type=str, default=None, help="Save fig location")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch Size")

    args = parser.parse_args()

    main(args)