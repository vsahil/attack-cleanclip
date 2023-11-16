import torch, os
from PIL import Image
import open_clip

cache_dir = "/gscratch/h2lab/vsahil/"
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
# os.environ['PYTHONUSERBASE'] = cache_dir
## change the python cache dir to store the model


# import ipdb; ipdb.set_trace()
# img = Image.open("data/ImageNet1K/validation/images/ILSVRC2012_val_00000236.JPEG")
# image = preprocess(img).unsqueeze(0).to("cuda:0")
# text = tokenizer(["a dog", "a cat", "a goldfish"]).to("cuda:0")

# import ipdb; ipdb.set_trace()

# with torch.no_grad(), torch.cuda.amp.autocast():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

from src.parser import parse_args
from src.data import get_train_dataloader
from tqdm import tqdm
os.environ["OMP_NUM_THREADS"] = "1" 
options = parse_args()

dataloader = get_train_dataloader(options, processor=None)

with torch.no_grad(), torch.cuda.amp.autocast():
    ## preprocess all the images in this batch, put in a list and then stack them.
    # import pandas as pd
    # df = pd.read_csv('../CC12M/training_data/backdoor_banana_random_random_16_100000_25.csv', sep=',')
    train_data_root = dataloader.dataset.root     # os.path.dirname('../CC12M/training_data/backdoor_banana_random_random_16_100000_25.csv')
    train_data_images = dataloader.dataset.images   #   df['image'].tolist()
    from PIL import Image

    if not "open_clip_image_features.pt" in os.listdir(".") or not "open_clip_text_features.pt" in os.listdir("."):
        image_features = []
        pretrained_model_name = 'ViT-SO400M-14-SigLIP-384'
        pretraining_dataset = 'webli'
        open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms(pretrained_model_name, pretrained=pretraining_dataset, cache_dir=cache_dir, device='cuda:0')
        open_clip_tokenizer = open_clip.get_tokenizer(pretrained_model_name)

        # import ipdb; ipdb.set_trace()
        for index, batch in tqdm(enumerate(dataloader)):
            # if options.inmodal:
            #     image_indexes = batch["original_idx"]      ## these are raw pixels, need to pass through the open_clip_preprocess
            # else:
            open_clip_image_processed = []
            image_indexes = batch["original_idx"].tolist()
            print("Starting batch: ", index, len(image_indexes))
            for image_idx in image_indexes:
                image = Image.open(os.path.join(train_data_root, train_data_images[image_idx]))
                image_here = open_clip_preprocess(image).unsqueeze(0)   #.to(options.device)
                open_clip_image_processed.append(image_here)
            open_clip_image_processed = torch.cat(open_clip_image_processed)
            assert open_clip_image_processed.shape[0] == len(image_indexes)
            image_features.append(open_clip_model.encode_image(open_clip_image_processed.to('cuda')))
            print("Done with batch: ", index)

        image_features = torch.cat(image_features)
        image_features /= image_features.norm(dim = -1, keepdim = True)
        ## open_clip_image_processed = torch.cat(open_clip_image_processed)
        ## image_features = open_clip_model.encode_image(open_clip_image_processed)
        torch.save(image_features, "open_clip_image_features.pt")
        
        # import ipdb; ipdb.set_trace()
        # config = eval(open(f"{options.eval_test_data_dir}/classes.py", "r").read())
        # classes, templates = config["classes"], config["templates"]
        # open_clip_text_embeddings = []
        # for c in tqdm(classes):
        #     text = open_clip_tokenizer([template(c) for template in templates]).to(options.device, non_blocking = True).to('cuda')
        #     text_embedding = open_clip_model.encode_text(text)
        #     text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
        #     text_embedding = text_embedding.mean(dim = 0)       ## take the mean of the 80 templates
        #     text_embedding /= text_embedding.norm()
        #     open_clip_text_embeddings.append(text_embedding)
        # open_clip_text_embeddings = torch.stack(open_clip_text_embeddings)
        
        # text_features = open_clip_text_embeddings
        # text_features /= text_features.norm(dim=-1, keepdim=True)

        # # save the text_features and image_features
        # torch.save(text_features, "open_clip_text_features.pt")
        # 

    else:
        image_features = torch.load("open_clip_image_features.pt")
        text_features = torch.load("open_clip_text_features.pt")
        import ipdb; ipdb.set_trace()    

        assert image_features.shape[0] == len(dataloader.dataset)
        assert text_features.shape[0] == 1000
        
        ## now get the class label for each image. Image and text features are already normalized.
        probabilties = (100.0 * image_features @ text_features.T)
        probabilties = probabilties.softmax(dim=-1)
        _, predicted = probabilties.max(1)
        assert len(predicted) == len(dataloader.dataset)
        import pandas as pd
        df = pd.DataFrame({'image': dataloader.dataset.images, 'idx': [i for i in range(len(dataloader.dataset))], 'label': predicted.tolist()})
        df.to_csv("open_clip_image_labels.csv", index=False)
