import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from backdoor.utils import ImageLabelDataset, ImageDataset
from pkgs.openai.clip import load as load_model
import numpy as np
import pandas as pd


def get_model(args, checkpoint):
    model, processor = load_model(name=args.model_name, pretrained=False)
    if (args.device == "cpu"): model.float()
    model.to(args.device)
    ckpt = torch.load(checkpoint, map_location=args.device)
    state_dict = ckpt["state_dict"]
    epoch = ckpt["epoch"]
    if (next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, processor, epoch


def clipscore(model, output):
    scores = model.logit_scale.exp() * output.image_embeds @ output.text_embeds.t()
    scores = torch.diagonal(scores)
    return scores


def get_count_per_fraction(all_examples_above_threshold, fraction):
    some_examples_above_threshold = all_examples_above_threshold[: int(fraction * len(all_examples_above_threshold))]
    actual_backdoor_examples_caught = 0
    for example in some_examples_above_threshold:
        if example[1]:
            actual_backdoor_examples_caught += 1
    results = {'total backdoors ': 300, 'backdoors detected ': actual_backdoor_examples_caught,
               'original detected as having backdoor ': len(
                   some_examples_above_threshold) - actual_backdoor_examples_caught}
    return results


def detect_backdoor_data(model, dataloader, args, epoch):
    with torch.no_grad():

        all_examples_above_threshold = []
        samples_detected_as_clean = []
        count_backdoored = 0
        count_total = 0

        for image, input_ids, attention_mask, is_backdoor, path, caption in tqdm(dataloader):
            count_backdoored += torch.sum(is_backdoor).detach().item()
            count_total += len(is_backdoor)
            image, input_ids, attention_mask = image.to(args.device), input_ids.to(args.device), attention_mask.to(
                args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=image)
            scores = clipscore(model, outputs)

            for index in range(len(scores)):
                if scores[index].item() > args.threshold:
                    all_examples_above_threshold.append((scores[index].item(), is_backdoor[index], path[index], caption[index]))
                else:
                    samples_detected_as_clean.append((path[index], caption[index]))

        all_examples_above_threshold.sort(key=lambda x: x[0], reverse=True)
        samples_detected_as_backdoored = all_examples_above_threshold[:int(args.fraction * len(all_examples_above_threshold))]
        number_of_backdoors_removed = np.sum([sample[1] for sample in samples_detected_as_backdoored])

        for example in all_examples_above_threshold[int(args.fraction * len(all_examples_above_threshold)):]:
            samples_detected_as_clean.append(example[-2:])

        print(f"total number of images in dataset = {count_total}\tnumber of images kept = {len(samples_detected_as_clean)}")
        print(f"total number of backdoored images = {count_backdoored}\tnumber of images removed = {len(samples_detected_as_backdoored)}\tnumber of backdoors removed = {number_of_backdoors_removed}")
        print(' ')

        print('this is what one entry of samples detected as clean looks like:')
        print(samples_detected_as_clean[0])
        print(' ')

        df_detected_as_backdoored = pd.DataFrame({'image': [example[-2] for example in samples_detected_as_backdoored], 'caption': [example[-1] for example in samples_detected_as_backdoored]})
        backdoored_write_path = args.original_csv.split('.')[0] + f'_detected_backdoored_{args.threshold}_{args.fraction}_epoch{epoch}' + '.' + args.original_csv.split('.')[1]
        print('backdoored write path: ', backdoored_write_path)
        print(' ')
        df_detected_as_backdoored.to_csv(backdoored_write_path, index=False)

        print('backdoored dataframe:')
        print(df_detected_as_backdoored.head())
        print(' ')

        df_detected_as_clean = pd.DataFrame({'image': [example[-2] for example in samples_detected_as_clean], 'caption': [example[-1] for example in samples_detected_as_clean]})
        clean_write_path = args.original_csv.split('.')[0] + f'_detected_clean_{args.threshold}_{args.fraction}_epoch{epoch}' + '.' + args.original_csv.split('.')[1]
        print('clean write path: ', clean_write_path)
        print(' ')
        df_detected_as_clean.to_csv(clean_write_path, index=False)

        print(all_examples_above_threshold[-1])
        print(get_count_per_fraction(all_examples_above_threshold, 0.01))
        print(get_count_per_fraction(all_examples_above_threshold, 0.02))
        print(get_count_per_fraction(all_examples_above_threshold, 0.05))
        print(get_count_per_fraction(all_examples_above_threshold, 0.1))
        print(get_count_per_fraction(all_examples_above_threshold, 0.15))
        print(get_count_per_fraction(all_examples_above_threshold, 0.20))
        print(get_count_per_fraction(all_examples_above_threshold, 0.25))
        print(get_count_per_fraction(all_examples_above_threshold, 0.30))
        print(get_count_per_fraction(all_examples_above_threshold, 0.35))
        print(get_count_per_fraction(all_examples_above_threshold, 0.40))
        print(get_count_per_fraction(all_examples_above_threshold, 0.45))
        print(get_count_per_fraction(all_examples_above_threshold, 0.50))


def main(args):
    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    model, processor, epoch = get_model(args, args.checkpoint)
    # dataset = ImageLabelDataset(args.original_csv, processor.process_image)
    dataset = ImageDataset(args.original_csv, processor, return_path=True, return_caption=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=16)
    # results = detect_backdoor(model, processor, dataloader, args)
    detect_backdoor_data(model, dataloader, args, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type=str, default=None, help="original csv with captions and images")
    parser.add_argument("--model_name", type=str, default="RN50", choices=["RN50", "RN101", "RN50x4", "ViT-B/32"],
                        help="Model Name")
    parser.add_argument("--device_id", type=str, default=None, help="device id")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/clip/", help="Path to checkpoint directories")
    parser.add_argument("--templates", type=str, default=None, help="classes py file")
    parser.add_argument("--batch_size", type=int, default=768, help="Batch Size")
    parser.add_argument("--patch_location", type=str, default="random", help="type of patch",
                        choices=["random", "four_corners", "blended"])
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for backdoor images")
    parser.add_argument("--threshold", type=float, default=40, help="Similarity threshold")
    parser.add_argument("--fraction", type=float, default=0.2,
                        help="How many top similarity samples to remove")

    args = parser.parse_args()
    main(args)
