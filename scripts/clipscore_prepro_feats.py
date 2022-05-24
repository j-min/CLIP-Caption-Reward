"""
Preprocess a raw json dataset into features files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: two folders of features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torch import nn

# preprocess = Compose([
#     Resize((448, 448), interpolation=Image.BICUBIC),
#     CenterCrop((448, 448)),
#     ToTensor()
# ])


# from clip.clip import load
# from timm.models.vision_transformer import resize_pos_embed
# import timm

# from captioning.utils.resnet_utils import myResnet
# import captioning.utils.resnet as resnet

from captioning.utils.clipscore import CLIPScore

from tqdm import tqdm



def main(params):

    clipscore_model = CLIPScore()
    clipscore_model.to('cuda')

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']

    if args.n_jobs > 1:
        print('Total imgs:', len(imgs))
        print('Using {} jobs'.format(args.n_jobs))
        print('job id:', args.job_id)
        imgs = imgs[args.job_id::args.n_jobs]

    N = len(imgs)

    seed(123) # make reproducible

    # dir_fc = params['output_dir']+'_clip_'+save_model_type+'_fc'
    # dir_att = params['output_dir']+'_clip_'+save_model_type+'_att'

    vis_dir_fc = params['output_dir']+'_clipscore_vis'
    if not os.path.isdir(vis_dir_fc):
        os.mkdir(vis_dir_fc)

    # text_dir_fc = params['output_dir']+'_clipscore_text'
    # if not os.path.isdir(text_dir_fc):
    #     os.mkdir(text_dir_fc)

    # if not os.path.isdir(dir_att):
    #     os.mkdir(dir_att)

    for i, img in enumerate(tqdm(imgs)):
        # load the image

        img_path = os.path.join(params['images_root'], img['filepath'], img['filename'])
        img_feat = clipscore_model.image_extract(img_path)
        img_feat = img_feat.view(512)

        # for d in img['sentences']:
        #     text = d['raw'].strip()
        #     text_feat = clipscore_model.text_extract(text)


        # with torch.no_grad():

            # image = preprocess(Image.open(os.path.join(params['images_root'], img['filepath'], img['filename']) ).convert("RGB"))
            # image = torch.tensor(np.stack([image])).cuda()
            # image -= mean
            # image /= std
            # if "RN" in params["model_type"]:
            #     tmp_att, tmp_fc = model.encode_image(image)
            #     tmp_att = tmp_att[0].permute(1, 2, 0)
            #     tmp_fc = tmp_fc[0]
            # elif params["model_type"] == 'vit_base_patch32_224_in21k':
            #     x = model(image)
            #     tmp_fc = x[0, 0, :]
            #     tmp_att = x[0, 1:, :].reshape( 14, 14, 768 )
            # else:
            #     x = model.visual.conv1(image.half())  # shape = [*, width, grid, grid]
            #     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            #     x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            #     x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            #     x = x + model.visual.positional_embedding.to(x.dtype)[:x.shape[1], :]
            #     x = model.visual.ln_pre(x)

            #     x = x.permute(1, 0, 2)  # NLD -> LND

            #     for layer_idx, layer in enumerate(model.visual.transformer.resblocks):
            #         x = layer(x)  

            #     x = x.permute(1, 0, 2)
            #     tmp_fc = x[0, 0, :]
            #     tmp_att = x[0, 1:, :].reshape( 14, 14, 768 )
                
        np.save(os.path.join(vis_dir_fc, str(img['cocoid'])), img_feat.data.cpu().float().numpy())
        # np.save(os.path.join(text_dir_fc, str(img['cocoid'])), tmp_fc.data.cpu().float().numpy())


        # np.savez_compressed(os.path.join(dir_att, str(img['cocoid'])), feat=tmp_att.data.cpu().float().numpy())

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
    print('wrote ', vis_dir_fc)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    # dataset_coco.json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='data', help='output h5 file')

    # options
    parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
    # parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
    # parser.add_argument('--model_type', default='RN50', type=str, help='RN50, RN101, RN50x4, ViT-B/32, vit_base_patch32_224_in21k')

    parser.add_argument('--n_jobs', default=-1, type=int, help='number of jobs to run in parallel')
    parser.add_argument('--job_id', default=0, type=int, help='job id')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
