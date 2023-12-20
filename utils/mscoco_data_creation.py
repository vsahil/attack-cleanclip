import os
import shutil
import json
import pandas as pd 
from tqdm import tqdm

def move_files(images, split):
    for image in tqdm(images):
        current_loc = os.path.join(root, 'val2014', image)
        dest_loc = os.path.join(root, split) 
        shutil.move(current_loc, dest_loc)    


def download_and_create_dataset(root_path):
    from clip_benchmark.datasets.builder import build_dataset
    ds = build_dataset("mscoco_captions", root=root_path, split="test") # this downloads the dataset if it is not there already -- I downloaded both train and test split. 
    coco = ds.coco
    imgs = coco.loadImgs(coco.getImgIds())
    future_df = {"filepath":[], "title":[]}
    for img in imgs:
        caps = coco.imgToAnns[img["id"]]
        for cap in caps:
            future_df["filepath"].append(img["file_name"])
            future_df["title"].append(cap["caption"])
    pd.DataFrame.from_dict(future_df).to_csv(os.path.join(root_path, "train2014.csv"), index=False, sep="\t")


def create_csv_files_from_karpathy_splits(root, split):
    with open('./data/karpathy_splits/dataset_coco.json') as f:
        dataset = json.load(f)
    test = list(filter(lambda x: x['split'] == split, dataset['images']))
    import ipdb; ipdb.set_trace()
    test_images = list(map(lambda x: x['filename'], test))
    test_captions = list(map(lambda x: x['sentences'][0]['raw'], test))
    # list_of_all_images = os.listdir(os.path.join(root, 'val2014'))
    # move_files(test_images, 'test')
    split_name = 'val' if split == 'test' else split
    test_images = list(map(lambda x: f'{split_name}/{x}', test_images))
    data = {'image': test_images, 'caption': test_captions}
    df = pd.DataFrame(data)
    df.to_csv(f'{root}/mscoco_{split}.csv')


if __name__ == "__main__":
    root = './data/MSCOCO'
    # download_and_create_dataset(root)
    create_csv_files_from_karpathy_splits(root, 'test')
    
