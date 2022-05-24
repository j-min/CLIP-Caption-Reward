from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
import json
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from torch.utils.data.distributed import DistributedSampler

from transformers import T5Tokenizer, BertTokenizer, BertTokenizerFast, CLIPTokenizer

import text_utils

project_dir = Path(__file__).parent.resolve()
workspace_dir = project_dir.parent.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
# coco_dir = dataset_dir.joinpath('COCO')
# vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = dataset_dir.joinpath('COCO/images/')
coco_data_dir = project_dir.parent.joinpath('CLIP-ViL/CLIP-ViL-Direct/caption/data/')
# coco_feature_dir = coco_dir.joinpath('features')


class COCORetrievalDataset(Dataset):
    def __init__(self, split='karpathy_train', rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.topk = topk
        self.verbose = verbose
        self.args = args
        self.rank = rank
        self.mode = mode

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)

        # if self.args.tokenizer is None:
        #     self.args.tokenizer = self.args.decoder_backbone

        # if 'bert' in self.args.tokenizer:
        #     self.tokenizer = BertTokenizerFast.from_pretrained(
        #         self.args.tokenizer,
        #         # max_length=self.args.max_text_length,
        #         # do_lower_case=self.args.do_lower_case
        #         )
        # elif 'clip' in self.args.tokenizer:
        #     self.tokenizer = CLIPTokenizer.from_pretrained(
        #         self.args.tokenizer,
        #         # max_length=self.args.max_text_length,
        #         # do_lower_case=self.args.do_lower_case
        #         )

        self.tokenizer = CLIPTokenizer.from_pretrained(
                self.args.tokenizer,
                # max_length=self.args.max_text_length,
                # do_lower_case=self.args.do_lower_case
                )

        with open(coco_data_dir.joinpath('cocotalk.json')) as f:
            self.vocab = list(json.load(f)['ix_to_word'].values())
            popped = self.vocab.pop(-1)
            assert popped == 'UNK'
            if self.verbose:
                print('vocab size: ', len(self.vocab))


        data_info_path = coco_data_dir.joinpath('dataset_coco.json')
        with open(data_info_path) as f:
            karpathy_data = json.load(f)

        split_rename = {
            'train': 'train',
            'restval': 'train',
            'val': 'val',
            'test': 'test'
        }

        n_images = 0

        data = []
        # self.vocab = set()
        for datum in karpathy_data['images']:
            re_split = split_rename[datum['split']]

            # if re_split == 'train':
            #     for d in datum['sentences']:
            #         self.vocab = self.vocab.union(set(d['tokens']))

            if re_split != self.source.split('_')[-1]:
                continue

            if re_split == 'train':
                # for d in datum['sentences']:
                #     img_id = datum['filename'].split('.')[0]
                #     new_datum = {
                #         'filename': datum['filename'],
                #         'img_id': img_id,
                #         'sent': d['raw'].strip(),
                #         'targets': [d['raw'].strip() for d in datum['sentences']],
                #         'is_train': True,
                #         'cocoid': datum['cocoid']
                #     }
                #     data.append(new_datum)
                img_id = datum['filename'].split('.')[0]
                new_datum = {
                    'filename': datum['filename'],
                    'img_id': img_id,
                    # 'sent': d['raw'],
                    # 'targets': [d['raw'].strip() for d in datum['sentences']],
                    'targets': [" ".join(d['tokens']) for d in datum['sentences']],
                    'is_train': True,
                    'cocoid': datum['cocoid']
                }
                data.append(new_datum)

            else:
                img_id = datum['filename'].split('.')[0]
                new_datum = {
                    'filename': datum['filename'],
                    'img_id': img_id,
                    # 'sent': d['raw'],
                    # 'targets': [d['raw'].strip() for d in datum['sentences']],
                    'targets': [" ".join(d['tokens']) for d in datum['sentences']],
                    'is_train': False,
                    'cocoid': datum['cocoid']
                }
                data.append(new_datum)

            n_images += 1

        if self.verbose:
            print(f"{self.source} has {n_images} images")
            # print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        # if self.verbose:
        #     print("# all sentences:", len(self.data))

        if self.args.load_feat:
            # feat_dir = coco_dir.joinpath(''
            # self.feat_loader = HybridLoader('/scratch-space/CLIP-ViL/CLIP-ViL-Direct/caption/data/cocotalk_clipscore_vis', ext='.npy', in_memory=False)
            self.feat_loader = HybridLoader(
                coco_data_dir.joinpath('cocotalk_clipscore_vis'),
                ext='.npy', in_memory=False)
        else:
            if 'openai/clip' in self.args.encoder_backbone:
                # from transformers import CLIPProcessor
                # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                #     size=args.image_size,
                #     do_resize=True,
                #     do_center_crop=False,
                # )
                # self.img_transform = lambda image: self.processor.feature_extractor(
                #     image,
                #     return_tensors='pt')['pixel_values'][0]

                self.image_mean = [0.48145466, 0.4578275, 0.40821073]
                self.image_std = [0.26862954, 0.26130258, 0.27577711]

                # captioning
                # self.img_transform = T.Compose([
                #     T.Resize((self.args.image_size, self.args.image_size))
                # ])

                # retrieval
                self.img_transform = T.Compose([
                    T.Resize(self.args.image_size, interpolation=T.functional.InterpolationMode.BICUBIC),
                    T.CenterCrop(self.args.image_size)
                ])

                self.img_tensor_transform = T.Compose([
                    # T.RandomCrop(224),
                    # T.RandomHorizontalFlip(p=0.3),
                    T.ConvertImageDtype(torch.float),
                    T.Normalize(self.image_mean, self.image_std)
                ]
                )
            # elif 'google/vit' in self.args.encoder_backbone:
            #     self.image_mean = [0.5, 0.5, 0.5]
            #     self.image_std = [0.5, 0.5, 0.5]

            #     self.img_transform = T.Compose([
            #         # T.PILToTensor(),
            #         T.Resize((self.args.image_size, self.args.image_size))
            #     ])

            #     self.img_tensor_transform = T.Compose([
            #         # T.RandomCrop(224),
            #         # T.RandomHorizontalFlip(p=0.3),
            #         T.ConvertImageDtype(torch.float),
            #         T.Normalize(self.image_mean, self.image_std)
            #     ]
            #     )

    def get_negative_text(self, text):
        neg_type = random.choice(['repeat', 'remove', 'insert', 'swap', 'shuffle'])

        if neg_type == 'repeat':
            text = text_utils.repeat(text)
        elif neg_type == 'remove':
            text = text_utils.remove(text)
        elif neg_type == 'insert':
            text = text_utils.insert(text, self.vocab)
        elif neg_type == 'swap':
            text = text_utils.swap(text, self.vocab)
        elif neg_type == 'shuffle':
            text = text_utils.shuffle(text)

        return text, neg_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        return self.process_datum(datum)

    def process_datum(self, datum):
        out_dict = {}

        ###### Image ######

        if self.args.load_feat:
            cocoid = datum['cocoid']
            out_dict['cocoid'] = str(cocoid)
            img_feat = self.feat_loader.get(str(cocoid))
            out_dict['img_feat'] = torch.from_numpy(img_feat)

        else:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id

            if 'train' in datum['filename']:
                img_split = 'train2014'
            elif 'val' in datum['filename']:
                img_split = 'val2014'
            img_path = coco_img_dir.joinpath(img_split).joinpath(datum['filename']).with_suffix('.jpg')
            assert img_path.exists()
            img_path = str(img_path)
            out_dict['img_path'] = img_path

            img_tensor = torchvision.io.read_image(img_path)
            # out_dict['img_tensor'] = img

            # img = Image.open(img_path).convert('RGB')
            # img_tensor = torch.as_tensor(np.asarray(img))
            out_dict['img_tensor'] = self.img_transform(img_tensor)
            # self.img_transform(img_tensor)
            # out_dict['img_tensor'] = self.img_transform(img)

        ###### Text #####
        # if datum['is_train']:
        # sent = datum['sent'].strip()

        sent = random.choice(datum['targets'])
        
        # target_ids = self.tokenizer.encode(
        #     sent, max_length=self.args.gen_max_length, truncation=True)

        # assert len(target_ids) <= self.args.gen_max_length, len(target_ids)
        out_dict['sent'] = sent
        # out_dict['target_ids'] = torch.LongTensor(target_ids)
        # out_dict['target_length'] = len(target_ids)


        # negative sample
        neg_sent, neg_type = self.get_negative_text(sent)

        # neg_target_ids = self.tokenizer.encode(
        #     neg_sent, max_length=self.args.gen_max_length, truncation=True)

        # assert len(neg_target_ids) <= self.args.gen_max_length, len(neg_target_ids)
        out_dict['neg_sent'] = neg_sent
        out_dict['neg_type'] = neg_type
        # out_dict['neg_target_ids'] = torch.LongTensor(neg_target_ids)
        # out_dict['neg_target_length'] = len(neg_target_ids)


        if 'targets' in datum:
            out_dict['targets'] = datum['targets']

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        # if 'target_ids' in batch[0]:
        #     T_W_L = max(entry['target_length'] for entry in batch)
        #     target_ids = torch.ones(
        #         B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        # if 'target_ids' in batch[0]:
        #     T_W_L = max(entry['target_length'] for entry in batch)
        #     target_ids = torch.ones(
        #         B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id



        targets = []
        img_ids = []
        img_paths = []

        coco_ids = []

        if self.args.load_feat:
            img_feats = torch.zeros(B, 512, dtype=torch.float)
        else:
            # imgs = []
            img_tensor = torch.zeros(B, 3, self.args.image_size, self.args.image_size, dtype=torch.uint8)

        for i, entry in enumerate(batch):

            if self.args.load_feat:
                coco_ids.append(entry['cocoid'])
                img_feats[i] = entry['img_feat']

            else:

                img_ids.append(entry['img_id'])
                img_paths.append(entry['img_path'])
                img_tensor[i] = entry['img_tensor']

            # if 'target_ids' in entry:
            #     target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'targets' in entry:
                targets.append(entry['targets'])

        if 'sent' in batch[0]:
            # word_mask = target_ids != self.tokenizer.pad_token_id
            # target_ids[~word_mask] = -100
            # batch_entry['target_ids'] = target_ids

            tokenized = self.tokenizer([entry['sent'] for entry in batch], truncation=True, padding=True, return_tensors='pt')
            neg_tokenized = self.tokenizer([entry['neg_sent'] for entry in batch], truncation=True, padding=True, return_tensors='pt')
                #     sent, max_length=self.args.gen_max_length, truncation=True)

            batch_entry['text'] = (tokenized.input_ids, tokenized.attention_mask)
            batch_entry['neg_text'] = (neg_tokenized.input_ids, neg_tokenized.attention_mask)


        if self.args.load_feat:
            batch_entry['coco_ids'] = coco_ids
            batch_entry['img_feats'] = img_feats

        else:

            img_tensor = self.img_tensor_transform(img_tensor)

            batch_entry['img_id'] = img_ids
            batch_entry['img_paths'] = img_paths
            batch_entry['img_tensor'] = img_tensor

        batch_entry['targets'] = targets

        # print('batch created')

        # batch_entry['task'] = 'caption'

        return batch_entry


# def get_loader(args, split='karpathy_train', mode='train',
#                batch_size=32, workers=4, distributed=False, gpu=0,
#                topk=-1):

#     verbose = (gpu == 0)

#     dataset = COCORetrievalDataset(
#         split,
#         rank=gpu,
#         topk=topk,
#         verbose=verbose,
#         args=args,
#         mode=mode)

#     # if distributed:
#     #     sampler = DistributedSampler(dataset)
#     # else:
#     #     sampler = None

#     if mode == 'train':
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=(sampler is None),
#             num_workers=workers, pin_memory=True, sampler=sampler,
#             collate_fn=dataset.collate_fn)
#     else:
#         loader = DataLoader(
#             dataset,
#             batch_size=batch_size, shuffle=False,
#             num_workers=workers, pin_memory=True,
#             sampler=sampler,
#             collate_fn=dataset.collate_fn,
#             drop_last=False)

#     # if verbose:
#         # loader.evaluator = COCOCaptionEvaluator()

#     # loader.task = 'caption'

#     return loader


# class COCOCaptionEvaluator:
#     def __init__(self):
#         import language_evaluation
#         self.evaluator = language_evaluation.CocoEvaluator(verbose=False)

#     def evaluate(self, predicts, answers):

#         results = self.evaluator.run_evaluation(predicts, answers)

#         return results

import six
import os
import h5py

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """

    def __init__(self, db_path, ext='.npy', in_memory=False):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            self.loader = lambda x: np.load(six.BytesIO(x))['feat']
        # if db_path.endswith('.lmdb'):
        #     self.db_type = 'lmdb'
        #     self.lmdb = lmdbdict(db_path, unsafe=True)
        #     self.lmdb._key_dumps = DUMPS_FUNC['ascii']
        #     self.lmdb._value_loads = LOADS_FUNC['identity']
        # elif db_path.endswith('.pth'):  # Assume a key,value dictionary
        #     self.db_type = 'pth'
        #     self.feat_file = torch.load(db_path)
        #     self.loader = lambda x: x
        #     print('HybridLoader: ext is ignored')
        # elif db_path.endswith('h5'):
        #     self.db_type = 'h5'
        #     self.loader = lambda x: np.array(x).astype('float32')
        # else:
        #     self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}

    def get(self, key):

        # if self.in_memory and key in self.features:
        #     # We save f_input because we want to save the
        #     # compressed bytes to save memory
        #     f_input = self.features[key]
        # elif self.db_type == 'lmdb':
        #     f_input = self.lmdb[key]
        # elif self.db_type == 'pth':
        #     f_input = self.feat_file[key]
        # elif self.db_type == 'h5':
        #     f_input = h5py.File(self.db_path, 'r')[key]
        # else:
            # f_input = open(os.path.join(
            #     self.db_path, key + self.ext), 'rb').read()

        f_input = open(os.path.join(
            self.db_path, key + self.ext), 'rb').read()

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input)

        return feat
