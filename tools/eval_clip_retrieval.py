
from PIL import Image
# import requests

from transformers import CLIPProcessor, CLIPModel

import torch
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from tqdm import tqdm
import json
import argparse
import numpy as np

class COCODataset(Dataset):
    def __init__(self,
                 coco_root="/nas-ssd/jmincho/datasets/COCO/",
                 gen_caption_path=None,
                 is_gt=True):
        super().__init__()

        self.coco_root = Path(coco_root)

        self.image_dir = self.coco_root.joinpath('images/val2014')

        if is_gt:
            print("Loading karpathy splits")
            data_info_path = self.coco_root.joinpath('dataset_coco.json')
            with open(data_info_path) as f:
                karpathy_data = json.load(f)

            data = []
            for datum in karpathy_data['images']:
                # karpathy test split
                if datum['split'] == 'test':
                    img_id = datum['filename'].split('.')[0]
                    new_datum = {
                        'img_id': img_id,
                        'captions': [d['raw'].strip() for d in datum['sentences']],
                    }
                    data.append(new_datum)
        else:
            print("Loading generated captions")
            gen_caption_path = Path(gen_caption_path)
            with open(gen_caption_path) as f:
                # karpathy_data = json.load(f)
                imgTogen_results = json.load(f)['imgToEval']
            data = []
            for img_id, img_data in imgTogen_results.items():
                new_datum = {
                    'img_id': img_id,
                    'captions': [img_data['caption']],
                }
                data.append(new_datum)

        self.data = data
        print('# images:', len(self.data))

        self.img_transform = processor.feature_extractor
        self.tokenizer = processor.tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        img_id = datum['img_id']
        if 'COCO' not in img_id:
            img_id = f'COCO_val2014_{str(img_id).zfill(12)}'
        img_fname = f"{img_id}.jpg"
        # COCO_val2014_000000522418.jpg
        img_path = self.image_dir.joinpath(img_fname)
        img = Image.open(img_path).convert("RGB")

        # take first caption
        caption = datum['captions'][0]

        return {
            "img": img,
            "caption": caption,
        }

    def collate_fn(self, datum_list):
        B = len(datum_list)
        imgs = [datum['img'] for datum in datum_list]
        images = self.img_transform(imgs, return_tensors="pt")

        captions  = [datum['caption'] for datum in datum_list]

        text_tokens = self.tokenizer(captions, return_tensors="pt", padding=True)
        batch = {
            'images': images,
            'captions': text_tokens,
        }
        return batch


def compute_similarity(image_features, text_features, bs = 1000):
    # compute similarity
    max_pairs = image_features.shape[0]
    similarity_scores = torch.zeros(max_pairs, max_pairs)
    for v in range(0, max_pairs, bs):
        for t in range(0, max_pairs, bs):
            # print('Processing Visual '+str(v)+' Text '+str(t), end='\r')
            batch_visual_emb = image_features[v:v+bs]
            batch_caption_emb = text_features[t:t+bs]

            logits = batch_visual_emb @ batch_caption_emb.t()
            similarity_scores[v:v+bs,t:t+bs] = logits

    print('Done similarity')
    return similarity_scores

def compute_retrieval(a2b_sims, return_ranks=True):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_root', type=str, default="/nas-ssd/jmincho/datasets/COCO/")
    parser.add_argument('--gt', action='store_true')
    parser.add_argument('--gen_caption_path', type=str, default="./eval_results/clipRN50_cider_test.json")
    args = parser.parse_args()

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = "cuda"
    model = model.to(device)
    model.eval()
    print(f"Loaded CLIP at {device}")

    batch_size = 1000

    dataset = COCODataset(
        coco_root="/nas-ssd/jmincho/datasets/COCO/",
        gen_caption_path=args.gen_caption_path,
        is_gt=args.gt
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        num_workers=8)

    # fwd all samples
    image_features = []
    text_features = []
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        # print('Evaluating batch {}/{}'.format(batch_idx, len(data_loader)), end="\r")
    #     images, texts = batch

        with torch.no_grad():
            images = batch["images"].to(device)
            texts = batch["captions"].to(device)

            vision_outputs = model.vision_model(**batch['images'])
            text_outputs = model.text_model(**batch['captions'])

            image_embeds = vision_outputs[1]
            image_embeds = model.visual_projection(image_embeds)

            text_embeds = text_outputs[1]
            text_embeds = model.text_projection(text_embeds)

            # normalized features
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        text_features.append(text_embeds.detach().cpu())
        image_features.append(image_embeds.detach().cpu())

    image_features = torch.cat(image_features, 0)
    text_features = torch.cat(text_features, 0)
    print('Done forward')

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # if not single_caption:
    #     for cap_idx in range(text_features.shape[1]):
    #         similarity_scores = compute_similarity(image_features, text_features[:,cap_idx,:])
    #         i2t_dict = compute_retrieval(similarity_scores.numpy())
    #         t2i_dict = compute_retrieval(similarity_scores.t().numpy())
    #         print(cap_idx, 'i2t', i2t_dict)
    #         print(cap_idx, 't2i', t2i_dict)
    # else:
    similarity_scores = compute_similarity(image_features, text_features)
    i2t_dict = compute_retrieval(similarity_scores.numpy())
    t2i_dict = compute_retrieval(similarity_scores.t().numpy())
    print('i2t', i2t_dict)
    print('t2i', t2i_dict)
