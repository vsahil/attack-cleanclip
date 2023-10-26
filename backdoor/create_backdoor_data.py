import os
import torch
import random
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
from backdoor.utils import apply_trigger
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

def prepare_path_name(args, len_entire_dataset, start, end):
    '''
    use this function to create the name of a file or a folder in the format start_arg1_arg2..._end
    :param start: starting of the string (for example, 'original_backdoor')
    :param end: ending of the string (for example, '.csv')
    '''

    output = start
    output += f'_{args.label}_{args.patch_type}_{args.patch_location}_{args.patch_size}'
    if args.size_train_data:
        output += f'_{args.size_train_data}'
    else:
        output += f'_{len_entire_dataset}'
    output += f'_{args.num_backdoor}'
    if args.label_consistent:
        output += '_label_consistent'
    output += end

    return output


def create_backdoor(args):
    # import ipdb; ipdb.set_trace()
    config    = eval(open(args.templates, "r").read())
    templates = config["templates"]

    root = os.path.dirname(args.train_data)
    # os.makedirs(os.path.join(root, "backdoor_images"), exist_ok = True)

    df   = pd.read_csv(args.train_data, sep = ',')

    indices = list(range(len(df)))
    len_entire_dataset = len(df)


    if args.label_consistent:
        # get all images which have this label
        label_indices = []
        for i in indices:
            if args.label in df.loc[i, 'caption']:
                label_indices.append(i)

        random.shuffle(label_indices)

        # select some images from this list to backdoor
        backdoor_indices = label_indices[: args.num_backdoor]

        # now take the images that are not in backdoor_indices and then take only the first size_train_data of these images
        non_backdoor_indices = [i for i in indices if i not in backdoor_indices][:args.size_train_data-args.num_backdoor]

    else:
        random.shuffle(indices)
        backdoor_indices = indices[: args.num_backdoor]
        non_backdoor_indices = indices[args.num_backdoor : args.size_train_data]

    df_backdoor = df.iloc[backdoor_indices, :]
    df_backdoor.to_csv(os.path.join(root, prepare_path_name(args, len_entire_dataset, 'original_backdoor', '.csv')))
    df_non_backdoor = df.iloc[non_backdoor_indices, :]
    
    locations, captions = [], []
    
    folder_name = prepare_path_name(args, len_entire_dataset, 'backdoor_images', '')
    os.makedirs(os.path.join(root, folder_name), exist_ok = True)

    for i in tqdm(range(len(df_backdoor))):

        image_loc  = df_backdoor.iloc[i]["image"]
        image_name = image_loc.split("/")[-1]

        image = Image.open(os.path.join(root, image_loc)).convert("RGB")
        image = apply_trigger(image, patch_size = args.patch_size, patch_type = args.patch_type, patch_location = args.patch_location)

        image_filename = f"{folder_name}/{image_name}"
        locations.append(image_filename)
        temp = random.randint(0, len(templates) - 1)

        if args.label_consistent:
            captions.append(df_backdoor.iloc[i]["caption"])

        if not args.label_consistent:
            captions.append(templates[temp](args.label))

        image.save(os.path.join(root, image_filename))

    data = {'image': locations, 'caption': captions}
    df_backdoor = pd.DataFrame(data)
    df = pd.concat([df_backdoor, df_non_backdoor])

    output_filename = prepare_path_name(args, len_entire_dataset, 'backdoor', '.csv')
    df.to_csv(os.path.join(root, output_filename))


def create_clean_dataset_file(args):
    ## the clean dataset is a subset from the backdoor file, excluding the backdoor images
    root = os.path.dirname(args.train_data)
    df   = pd.read_csv(args.train_data, sep = ',')
    len_entire_dataset = len(df)
    df = pd.read_csv(os.path.join(root, prepare_path_name(args, len_entire_dataset, 'backdoor', '.csv')))
    ## assert that the backdoor images are at the beginning of the file. Take the first num_backdoor images, and see if the label in in the caption
    for i in range(args.num_backdoor):
        assert args.label in df.loc[i, 'caption']
    
    ## take any random 100K images from lines that is not in the first num_backdoor lines
    indices = list(range(args.num_backdoor, len(df)))
    random.shuffle(indices)
    indices = indices[:args.size_clean_data]
    df_clean = df.iloc[indices, :]
    df_clean.to_csv(os.path.join(root, prepare_path_name(args, len_entire_dataset, 'clean', '.csv')), index=False)


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--label", type = str, default = "banana", help = "Label to backdoor")
    parser.add_argument("--templates", type = str, default = None, help = "classes py file")
    parser.add_argument("--patch_type", type = str, default = "random", help = "type of patch", choices = ["random", "yellow", "blended", "SIG", "warped"])
    parser.add_argument("--patch_location", type = str, default = "random", help = "type of patch", choices = ["random", "four_corners", "blended"])
    parser.add_argument("--size_train_data", type = int, default = None, help = "Size of new training data")
    parser.add_argument("--size_clean_data", type = int, default = None, help = "Size of new clean training data")
    parser.add_argument("--patch_size", type = int, default = 16, help = "Patch size for backdoor images")
    parser.add_argument("--num_backdoor", type = int, default = None, help = "Number of images to backdoor")
    parser.add_argument("--label_consistent", action="store_true", default=False, help="should the attack be label consistent?")

    args = parser.parse_args()
    # import ipdb; ipdb.set_trace()
    if args.size_clean_data is None:
        create_backdoor(args)
    else:
        create_clean_dataset_file(args)


'''
Run Example:

python -m backdoor.create_backdoor_data --train_data /data0/CC3M/train/train.csv  --templates /data0/datasets/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 300 --patch_type blended --patch_location blended
'''
