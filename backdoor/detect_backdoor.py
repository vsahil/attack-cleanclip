import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from backdoor.utils import ImageLabelDataset, ImageDataset
from pkgs.openai.clip import load as load_model


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

# def get_count_backdoor(image_embedding, text_embeddings, threshold):
#     logits = (image_embedding @ text_embeddings)
#     max_values, max_indices = logits.max(dim = 1)
#     backdoored = (max_values > threshold)
#     count = backdoored.sum().item()
#     return max_values, max_values, count

def clipscore(model, output):
    scores = model.logit_scale.exp() * output.image_embeds @ output.text_embeds.t()
    scores = torch.diagonal(scores)
    return scores

def get_count_per_fraction(all_examples_above_threshold, fraction):
    some_examples_above_threshold   = all_examples_above_threshold[: int(fraction * len(all_examples_above_threshold))]
    actual_backdoor_examples_caught = 0
    for example in some_examples_above_threshold:
        if example[1]:
                actual_backdoor_examples_caught += 1
    results = {'total backdoors ': 300, 'backdoors detected ': actual_backdoor_examples_caught, 'original detected as having backdoor ': len(some_examples_above_threshold) - actual_backdoor_examples_caught}
    return results

def detect_backdoor_data(model, dataloader, args):
    with torch.no_grad():
        count_detected = 0
        count_original_detected = 0
        total_backdoor = 0
        all_examples_above_threshold = []
        for image, input_ids, attention_mask, is_backdoor in tqdm(dataloader):
            image, input_ids, attention_mask = image.to(args.device), input_ids.to(args.device), attention_mask.to(args.device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = image)
            scores = clipscore(model, outputs)
            total_backdoor += is_backdoor.sum()
            for index in range(len(is_backdoor)):
                if is_backdoor[index]:
                    count_detected += (scores[index] > args.threshold)
                else:
                    count_original_detected += (scores[index] > args.threshold)
            for index in range(len(scores)):
                if scores[index].item() > args.threshold:
                    all_examples_above_threshold.append((scores[index].item(), is_backdoor[index]))
        all_examples_above_threshold.sort(key = lambda x: x[0], reverse = True)
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

# def detect_backdoor(model, processor, dataloader, args):

#     config = eval(open(f"{args.templates}", "r").read())
#     classes, templates = config["classes"], config["templates"]

#     with torch.no_grad():
#         text_embeddings = []
#         for c in tqdm(classes):
#             text = [template(c) for template in templates]
#             text_tokens = processor.process_text(text)
#             text_input_ids, text_attention_mask = text_tokens["input_ids"].to(args.device), text_tokens["attention_mask"].to(args.device) 
#             text_embedding = model.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
#             text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
#             text_embedding = text_embedding.mean(dim = 0)
#             text_embedding /= text_embedding.norm()
#             text_embeddings.append(text_embedding)
#         text_embeddings = torch.stack(text_embeddings, dim = 1).to(args.device)

#         count_original, count_backdoor = 0, 0
#         total = 0
#         collection_max_original, collection_max_backdoor = [], []
#         for image, image_backdoor, label in tqdm(dataloader):

#             image, image_backdoor, label = image.to(args.device), image_backdoor.to(args.device), label.to(args.device)
            
#             image_embedding = model.get_image_features(image)
#             image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
#             image_backdoor_embedding = model.get_image_features(image_backdoor)
#             image_backdoor_embedding /= image_backdoor_embedding.norm(dim = -1, keepdim = True)

#             max_values_original, max_indices_original, count_original = get_count_backdoor(image_embedding, text_embeddings, args.threshold)
#             collection_max_original.append(max_values_original)    
#             count_original += count_original

#             max_values_backdoor, max_indices_backdoor, count_backdoor = get_count_backdoor(image_backdoor_embedding, text_embeddings, args.threshold)
#             collection_max_backdoor.append(max_values_backdoor)    
#             count_backdoor += count_backdoor

#             total += len(label)

#         collection_max_original = torch.cat(collection_max_original, dim = 0)
#         average_original, std_original = collection_max_original.mean().item(), collection_max_original.std().item() 
#         percent_original = 100 * count_original / total

#         collection_max_backdoor = torch.cat(collection_max_backdoor, dim = 0)
#         average_backdoor, std_backdoor = collection_max_backdoor.mean().item(), collection_max_backdoor.std().item() 
#         percent_backdoor = 100 * count_backdoor / total

#         results = {'Percent of examples detected with backdoor (O)' : percent_original, 'Mean of max (O)': average_original, 'Std of max (O)': std_original, \
#                     'Percent of examples detected with backdoor (B)' : percent_backdoor, 'Mean of max (B)': average_backdoor, 'Std of max (B)': std_backdoor,}
#         return results

def main(args):
    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")    
    model, processor = get_model(args, args.checkpoint)
    # dataset = ImageLabelDataset(args.original_csv, processor.process_image)
    dataset = ImageDataset(args.original_csv, processor)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False, num_workers=8)
    # results = detect_backdoor(model, processor, dataloader, args)
    detect_backdoor_data(model, dataloader, args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--device_id", type = str, default = None, help = "device id")
    parser.add_argument("--checkpoint", type = str, default = "checkpoints/clip/", help = "Path to checkpoint directories")
    parser.add_argument("--templates", type = str, default = None, help = "classes py file")
    parser.add_argument("--batch_size", type = int, default = 768, help = "Batch Size")
    parser.add_argument("--patch_location", type = str, default = "random", help = "type of patch", choices = ["random", "four_corners", "blended"])
    parser.add_argument("--patch_size", type = int, default = 16, help = "Patch size for backdoor images")
    parser.add_argument("--threshold", type = float, default = 40, help = "Similarity threshold")
    parser.add_argument("--fraction", type = float, default = 0.2, help = "How many of total backdoors should be considered here")

    args = parser.parse_args()
    main(args)
