import argparse
import os, json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from PIL import Image, ImageFile
from pkgs.openai.clip import load
from backdoor.utils import apply_trigger as apply_trigger_images

ImageFile.LOAD_TRUNCATED_IMAGES = True

def batch(iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


@torch.no_grad()
def itm_eval(text_embeddings, image_embeddings, options, df=None, target_label_closest_indices=None):
    ## Revised code for Image -> Text retrieval, CleanCLIP code between the image and text retrieval was switched in the wrong direction. 
    similarity_matrix = (image_embeddings @ text_embeddings.T).cpu().numpy()

    # Get the indices that would sort the matrix along each row in descending order
    sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]

    if not options.add_backdoor:        ## This means we are doing the evaluation over the clean MSCOCO dataset. 
        # Find the rank of the true matches
        true_indices = np.arange(len(image_embeddings))
        ranks = (sorted_indices == true_indices[:, None]).argmax(axis=1)
        # Compute metrics
        ir1 = 100.0 * np.mean(ranks < 1)
        ir5 = 100.0 * np.mean(ranks < 5)
        ir10 = 100.0 * np.mean(ranks < 10)
        eval_result = {'img_to_text_recall_at_1': ir1, 'img_to_text_recall_at_5': ir5, 'img_to_text_recall_at_10': ir10, 'img_to_text_recall_mean': (ir1 + ir5 + ir10)/3}
    else:
        # import ipdb; ipdb.set_trace()
        ## Here we do some complex stuff. We get the top-k images for each text, and then check if majority of them belong to the target label set indices. Target label set indices are all indices of MSCOCO test dataset that have the word "banana" in their captions. Only if the majority of the top-k images belong to the target label set, we consider it a success for the backdoor attack. 
        ## These are the indices in the test set of MSCOCO that have "banana" in their captions.
        if options.input_file_name == "mscoco_test": #and not options.use_semantic_closest_captions:
            target_label_indices = [40, 67, 69, 70, 210, 334, 419, 590, 592, 593, 713, 714, 715, 842, 1080, 1137, 1199, 1424, 1545, 1546, 1612, 1793, 1797, 1938, 2059, 2117, 2310, 2561, 2634, 2952, 2954, 3072, 3496, 3497, 4026, 4028, 4221, 4536, 4767, 4768, 4972]
        elif options.input_file_name == "cc6m_small": # and not options.use_semantic_closest_captions:
            target_label_indices = [90, 179, 618, 689, 694, 807, 874, 932, 1162, 1219, 1244, 1447, 1536, 1600, 1767, 1922, 1948, 1949, 2167, 2188, 2383, 2400, 2488, 2696, 2767, 2821, 2902, 2979, 3134, 3295, 3350, 3375, 3387, 3424, 3712, 3715, 3750, 3881, 3957, 4356, 4471, 4523, 4772]
        if options.use_semantic_closest_captions:
            ## merge the target_label_indices from the options with the target_label_indices from the function find_closest_captions_for_target_label_semantically
            target_label_indices = target_label_closest_indices + target_label_indices
        target_label_indices = set(target_label_indices)

        print(target_label_indices, len(target_label_indices))
        ## Now we have the sorted indices, we need to get the top-k texts for each image, and see if majority of them belong to the target label set. 
        eval_result = {}
        for topk in [1, 5, 10]:
            correct = 0
            for i in range(len(image_embeddings)):
                topk_indices = sorted_indices[i][:topk]
                # if topk == 5:
                #     print(f'Image: {i}, Top-5: {topk_indices}, Predicted: {[df.iloc[j]["caption"] for j in topk_indices]}\n')
                if len(set(topk_indices).intersection(target_label_indices)) >= topk//2 and len(set(topk_indices).intersection(target_label_indices)) > 0:       ## this should always be greater than 0.
                    correct += 1
                elif options.print_datapoints_not_target_top1 and len(set(topk_indices).intersection(target_label_indices)) == 0 and (topk == 1):
                    print(f'Image: {i}, Top-1: {topk_indices[0]}, Predicted: {df.iloc[topk_indices[0]]["caption"]}, Actual: {df.iloc[i]["caption"]}\n')
            
            print(f'Top-{topk} Retrieval ASR: {correct * 100. /len(image_embeddings)}')
            eval_result[f'img_to_text_recall_at_{topk}'] = correct * 100. /len(image_embeddings)

        eval_result['img_to_text_recall_mean'] = (eval_result['img_to_text_recall_at_1'] + eval_result['img_to_text_recall_at_5'] + eval_result['img_to_text_recall_at_10'])/3
    return eval_result


@torch.no_grad()
def find_closest_captions_for_target_label_semantically(model, processor, text_embeds, device, closest_k, target_word):
    config = eval(open(f"/gscratch/cse/vsahil/attack-cleanclip/data/ImageNet1K/validation/classes.py", "r").read())
    templates = config["templates"]
    
    for c in [target_word]:
        text = [template(c) for template in templates]
        assert len(text) == 80
        text_tokens = processor.process_text(text)
        text_input_ids, text_attention_mask = text_tokens["input_ids"].to(device), text_tokens["attention_mask"].to(device) 
        text_embedding = model.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
        text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
        text_embedding = text_embedding.mean(dim = 0)
        text_embedding /= text_embedding.norm()
    target_text_embedding = text_embedding.reshape(1, -1)

    # import ipdb; ipdb.set_trace()
    ## find the closest k captions to the target_text_embedding
    similarity_matrix = (text_embeds @ target_text_embedding.T).cpu().numpy()
    sorted_indices = np.argsort(similarity_matrix, axis=0)[::-1]
    target_label_indices = sorted_indices[:closest_k]
    return target_label_indices.flatten().tolist()


def get_all_embeddings(options, model, all_texts, all_images, root, processor, batch_size = 1024, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = False):
    text_embeddings = []
    image_embeddings = []

    with torch.no_grad():
        score = 0

        dataloader_texts = list(batch(all_texts, batch_size))
        dataloader_images = list(batch(all_images, batch_size))

        bar = zip(dataloader_texts, dataloader_images)
        print("Evaluating..")
        bar = tqdm(bar, total = len(dataloader_texts))
        
        for texts, images in bar:
            captions = processor.process_text(texts)
            input_ids = captions['input_ids'].to(device)
            attention_mask = captions['attention_mask'].to(device)
        
            if not options.add_backdoor:   
                pixel_values = torch.tensor(np.stack([processor.process_image(Image.open(os.path.join(root, image)).convert("RGB")) for image in images])).to(device)       ## This is the line that loads the images. 
            else:
                pixel_values = []
                for image in images:
                    image = Image.open(os.path.join(root, image)).convert("RGB")
                    image = apply_trigger_images(image, patch_size = options.patch_size, patch_type = options.patch_type, patch_location = options.patch_location)
                    image = processor.process_image(image)
                    pixel_values.append(image)
                pixel_values = torch.tensor(np.stack(pixel_values)).to(device)
            
            text_embedding = model.get_text_features(input_ids = input_ids, attention_mask = attention_mask)
            image_embedding = model.get_image_features(pixel_values)

            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)

            text_embeddings.append(text_embedding)
            image_embeddings.append(image_embedding)

        text_embeddings = torch.cat(text_embeddings)
        image_embeddings = torch.cat(image_embeddings)
        return text_embeddings, image_embeddings


def evaluate(options):
    print(options.input_file)
    root = os.path.dirname(options.input_file)
    df = pd.read_csv(options.input_file, sep = options.delimiter)

    if os.path.exists(options.embeddings_file) and not "banana" in options.input_file and not options.use_semantic_closest_captions:      ## when we have replaced the banana containing captions with just banana, we aren't going to save the embeddings.
        with open(options.embeddings_file, 'rb') as f:
            text_embeds, image_embeds = pickle.load(f)
        print('Embeddings Loaded!')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = load(name = options.model_name, pretrained = options.pretrained)
        model = model.to(device)
        if(options.checkpoint is not None):
            if(os.path.isfile(options.checkpoint)):
                checkpoint = torch.load(options.checkpoint, map_location = device)
                state_dict = checkpoint['state_dict']
                if(next(iter(state_dict.items()))[0].startswith("module")):
                    state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
                model.load_state_dict(state_dict)
                print(f'Loaded checkpoint {options.checkpoint}')
            else:
                print(f'No checkpoint found at {options.checkpoint}')

        model.eval()

        captions = df[options.caption_key].tolist()
        images = df[options.image_key].tolist()

        if os.path.exists(options.embeddings_file) and not "banana" in options.input_file:      ## This is for the case when options.use_semantic_closest_captions is True, and we have already saved the embeddings.
            with open(options.embeddings_file, 'rb') as f:
                text_embeds, image_embeds = pickle.load(f)
            print('Embeddings Loaded!')
        else:
            text_embeds, image_embeds = get_all_embeddings(options, model, captions, images, root = root, processor = processor, batch_size = options.batch_size, device = device)
        
        if options.use_semantic_closest_captions:
            target_label_indices = find_closest_captions_for_target_label_semantically(model, processor, text_embeds, device, closest_k=options.closest_k_semantic, target_word='banana')

        if not "banana" in options.input_file:      ## when we have replaced the banana containing captions with just banana, we aren't going to save the embeddings.
            with open(options.embeddings_file, 'wb') as f:
                pickle.dump((text_embeds, image_embeds), f)
            print('Image and text embeddings dumped!')

    result = itm_eval(text_embeds, image_embeds, options, df, target_label_indices if options.use_semantic_closest_captions else None)

    if options.print_datapoints_not_target_top1:
        return
    ## write the results to the results file, as a dictionary that has the options and the results as a json. Make sure to append the results
    results_file = f'/gscratch/cse/vsahil/attack-cleanclip/utils/retrieval_results/all_models_retrieval_results_{options.input_file_name}.csv'
    ## the first column will be the model pretraining, the second column will be the pre-training dataset, third column will be boolean value of add_backdoor, and 4, 5, 6, and 7th columns will be the retrieval results. Everthing will be tab separated.
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write('model_pretraining\tpretraining_dataset\tadd_backdoor\timg_to_text_recall_at_1\timg_to_text_recall_at_5\timg_to_text_recall_at_10\timg_to_text_recall_mean\n')
    with open(results_file, 'a') as f:
        f.write(f'{options.model_pretraining}\t{options.pretraining_dataset}\t{options.add_backdoor}\t{result["img_to_text_recall_at_1"]}\t{result["img_to_text_recall_at_5"]}\t{result["img_to_text_recall_at_10"]}\t{result["img_to_text_recall_mean"]}\n')

    print(result)
    print('Results also written to file!')


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file", type = str, default = None, help = "Input file", required=True)
    parser.add_argument("--batch_size", type = int, default = 1024, help = "Batch Size")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image", help = "Image column name")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "Caption column name")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    
    # parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models. For our case pretrained will be False as we want to evaluate our trained models. ")
    # parser.add_argument("--use_saved_embeddings", action = "store_true", default = False, help = "Use saved embeddings")
    # parser.add_argument("--embeddings_file", type = str, default = "embeddings.pkl", help = "embedding file")
    
    ## This is for the case when we want to add backdoor to the images, and then get the retrieval results. Let's see if there is any banana in the test set of MsCOCO -- yes there are about 42 bananas, so they should match. 
    parser.add_argument("--add_backdoor", default = False, action = "store_true", help = "add backdoor or not")
    parser.add_argument("--patch_type", default = None, type = str, help = "patch type of backdoor")
    parser.add_argument("--patch_location", default = None, type = str, help = "patch location of backdoor")
    parser.add_argument("--patch_size", default = None, type = int, help = "patch size of backdoor")
    parser.add_argument("--use_semantic_closest_captions", default = False, action = "store_true", help = "use semantic closest captions. If not, we use the captions that contain the word banana - text matching. ")
    parser.add_argument("--closest_k_semantic", default = 20, type = int, help = "closest k captions to find semantically for the target label. ")

    parser.add_argument("--print_datapoints_not_target_top1", default = False, action = "store_true", help = "print datapoints that have a trigger added but the predicted top-1 caption does not have banana")

    ## we will use this to determine which model we need to load. 
    parser.add_argument("--model_pretraining", type = str, default = None, choices = ["mmcl", "mmcl_ssl"], required=True)
    parser.add_argument("--pretraining_dataset", type=str, default=None, choices=["cc6m", "cc3m"], required=True)

    options = parser.parse_args()

    if options.pretraining_dataset == 'cc6m':
        if options.model_pretraining == 'mmcl':       ## experiments with CC6M
            checkpoint = '/gscratch/cse/vsahil/attack-cleanclip/logs/train_cc6m_poison_mmcl_1e_3/checkpoints/epoch_21.pt'
        elif options.model_pretraining == 'mmcl_ssl':
            checkpoint = '/gscratch/cse/vsahil/attack-cleanclip/logs/train_cc6m_poison_mmcl_ssl_1e_3_batch1024/checkpoints/epoch_36.pt'
        else:
            raise NotImplementedError
    elif options.pretraining_dataset == 'cc3m':
        if options.model_pretraining == 'mmcl':       ## experiments with CC6M
            checkpoint = '/gscratch/cse/vsahil/attack-cleanclip/logs/train_1_poison_mmcl/checkpoints/epoch.best.pt'
        elif options.model_pretraining == 'mmcl_ssl':
            checkpoint = '/gscratch/cse/vsahil/attack-cleanclip/logs/train_newa100_poison_mmcl_ssl_both_1e_3_lr/checkpoints/epoch.best.pt'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    options.checkpoint = checkpoint
    if "mscoco_test" in options.input_file:
        options.input_file_name = "mscoco_test"
    elif "cc6m_small" in options.input_file:
        options.input_file_name = "cc6m_small"
    else:
        raise NotImplementedError
    options.embeddings_file = f'/gscratch/cse/vsahil/attack-cleanclip/utils/retrieval_results/embeddings_{options.model_pretraining}_{options.pretraining_dataset}_{"poisoned" if options.add_backdoor else "clean"}_{options.input_file_name}.pkl'
    
    evaluate(options)


'''
In this file, --asr is not needed as it is for ImageNet accuracy because for the accuracy case we are doing the clean dataset evaluation and poisoned dataset evaluation in the same run. Whereas in this case, add_backdoor dictates whether we are doing evaluation on clean or poisoned dataset. 
Command to run this file: python -m utils.retrieval --input_file utils/data/MSCOCO/mscoco_test.csv --model_pretraining 'mmcl_ssl' --pretraining_dataset 'cc3m'          ## for clean dataset
or 
python -m utils.retrieval --input_file utils/data/MSCOCO/mscoco_test.csv --model_pretraining 'mmcl' --pretraining_dataset 'cc3m' --add_backdoor --patch_type random  --patch_location random --patch_size 16        ## for poisoned dataset
'''


