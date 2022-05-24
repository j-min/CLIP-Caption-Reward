from transformers import CLIPModel, CLIPTokenizer
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


class CLIPScore(nn.Module):
    def __init__(self, clipscore_w=2.5, image_size=224, mode='clip_s', use_grammar=False, joint_out=False):
        super(CLIPScore, self).__init__()
        # from transformers import CLIPModel, CLIPTokenizer
        self.clip_model = CLIPModel.from_pretrained(
            'openai/clip-vit-base-patch32')
        self.tokenizer = CLIPTokenizer.from_pretrained(
            'openai/clip-vit-base-patch32')

        self.clip_model.eval()

        self.clipscore_w = clipscore_w

        self.image_transform = self._transform(image_size)

        self.mode = mode
        assert mode in ['clip_s', 'refclip_s']

        self.use_grammar = use_grammar
        self.joint_out = joint_out

        if self.use_grammar and joint_out is False:
            self.grammar_score_head = nn.Sequential(
                nn.Linear(self.clip_model.text_embed_dim, self.clip_model.projection_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.clip_model.projection_dim, 2, bias=False)
            )

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def load_image(self, image_path):
        image = Image.open(image_path)
        return image

    # @torch.no_grad()
    def image_extract(self, image):
        if isinstance(image, str):
            image = self.load_image(image)
        if not isinstance(image, torch.Tensor):
            image = self.image_transform(image)

        img_tensor = image.view(-1, 3, 224, 224)
        device = next(self.clip_model.parameters()).device
        img_tensor = img_tensor.to(device)

        clip_model = self.clip_model

        img_feat = clip_model.vision_model(img_tensor).pooler_output
        img_feat = clip_model.visual_projection(img_feat)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        return img_feat

    # @torch.no_grad()
    def text_extract(self, text, prompt="A photo depicts", proj_norm=True):
        if isinstance(text, str):
            text_batch = [" ".join([prompt, text])]
        elif isinstance(text, list):
            text_batch = [" ".join([prompt, txt]) for txt in text]
        
        if isinstance(text, tuple) and isinstance(text[0], torch.Tensor):
            input_ids, attention_mask = text
        else:
            input_text = text_batch

            tokenized = self.tokenizer(
                input_text, return_tensors='pt', padding=True, truncation=True)

            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask

        clip_model = self.clip_model
        device = next(self.clip_model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        text_feat = clip_model.text_model(input_ids, attention_mask).pooler_output

        if proj_norm:
            text_feat = clip_model.text_projection(text_feat)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        return text_feat

    # @torch.no_grad()
    def calc_clip_s(self, img_feat, text_feat):
        return self.clipscore_w * torch.relu((img_feat * text_feat).sum(dim=-1))

    # @torch.no_grad()
    def calc_refclip_s(self, img_feat=None, text_feat=None, ref_text_feat=None, ref_text_mask=None, clip_s=None):

        if clip_s is None:
            clip_s = self.calc_clip_s(img_feat, text_feat)

        B, dim = img_feat.size()

        ref_text_feat = ref_text_feat.view(B, -1, dim)

        K = ref_text_feat.size(1)

        text_feat = text_feat.view(B, 1, dim).expand(-1, K, -1)
        assert ref_text_feat.size() == text_feat.size(
        ), (ref_text_feat.size(), text_feat.size())

        ref_score = self.calc_clip_s(text_feat, ref_text_feat)
        if ref_text_mask is not None:
            if not isinstance(ref_text_mask, torch.Tensor):
                ref_text_mask = torch.tensor(
                    ref_text_mask, dtype=ref_score.dtype, device=ref_score.device)
            ref_score = ref_score.view(B, K) * ref_text_mask.view(B, K)

        ref_score = ref_score.view(B, K).max(dim=1).values

        assert clip_s.size() == (B,)
        assert clip_s.size() == ref_score.size()

        # harmonic mean
        refclip_s = 2 / (1 / clip_s + 1 / ref_score)
        return refclip_s

    @torch.no_grad()
    def forward(self,
                images=None, text=None,
                img_feat=None, text_feat=None,
                ref_text=None, ref_text_feat=None, ref_text_mask=None,
                prompt="A photo depicts",
                mode=None):
        if img_feat is None:
            img_feat = self.image_extract(images)
        img_feat = img_feat.view(-1, 512)

        B = img_feat.size(0)

        if text_feat is None:
            text_feat = self.text_extract(text, prompt=prompt)
        text_feat = text_feat.view(-1, 512)

        if mode is None:
            mode = self.mode
        assert mode in ['clip_s', 'refclip_s']

        if mode == 'clip_s':
            clip_s = self.calc_clip_s(img_feat, text_feat)
            return clip_s
        elif mode == 'refclip_s':
            if ref_text_feat is None:
                ref_text_feat = self.text_extract(ref_text, prompt=prompt)
            ref_text_feat = ref_text_feat.view(-1, 512)

            refclip_s = self.calc_refclip_s(
                img_feat, text_feat, ref_text_feat, ref_text_mask=ref_text_mask)
            return refclip_s


    def train_step(self,
                   images=None, text=None,
                   img_feat=None, text_feat=None,
                   neg_text=None, neg_text_feat=None,
                #    ref_text=None, ref_text_feat=None, ref_text_mask=None,
                   prompt="A photo depicts",
                #    return_loss=True,
                   **kwargs):

        if img_feat is None:
            img_feat = self.image_extract(images)
        img_feat = img_feat.view(-1, 512)

        B = img_feat.size(0)

        if text_feat is None:
            text_feat = self.text_extract(text, prompt=prompt, proj_norm=False)

            text_cont_feat = self.clip_model.text_projection(text_feat)
            text_cont_feat = text_cont_feat / text_cont_feat.norm(dim=-1, keepdim=True)
        text_cont_feat = text_cont_feat.view(B, 512)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_cont_feat, img_feat.t()) * logit_scale
        # logits_per_image = logits_per_text.T

        clip_loss = clip_loss_fn(logits_per_text)


        # negative sampling
        pos_text_feat = text_feat.view(B, 512)
        neg_text_feat = self.text_extract(neg_text, prompt=prompt, proj_norm=False).view(B, 512)

        grammar_text_feat = torch.cat([pos_text_feat, neg_text_feat], dim=0)

        # 2B, 1
        grammar_text_logit = self.grammar_score_head(grammar_text_feat)
        grammar_labels = torch.LongTensor([1] * B + [0] * B).to(grammar_text_logit.device).view(2 * B)

        grammar_loss = torch.nn.functional.cross_entropy(grammar_text_logit, grammar_labels)

        grammar_pred = grammar_text_logit.argmax(dim=1, keepdim=False)
        grammar_pos_pred = grammar_pred[:B]
        grammar_neg_pred = grammar_pred[B:]
        # grammar_acc = (grammar_pred == grammar_labels).float().mean()

        out = {
            'clip_loss': clip_loss,
            'grammar_loss': grammar_loss,
            'img_feat': img_feat,
            'text_feat': text_cont_feat,
            'neg_text_feat': neg_text_feat,
            'grammar_pos_pred': grammar_pos_pred,
            'grammar_neg_pred': grammar_neg_pred,
        }

        return out

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor, dim: int) -> torch.Tensor:
    neg_ce = torch.diag(nn.functional.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def clip_loss_fn(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0



# class CLIPScore(nn.Module):
#     def __init__(self, clipscore_w=2.5, image_size=224, mode='clip_s'):
#         super(CLIPScore, self).__init__()
#         # from transformers import CLIPModel, CLIPTokenizer
#         self.clip_model = CLIPModel.from_pretrained(
#             'openai/clip-vit-base-patch32')
#         self.tokenizer = CLIPTokenizer.from_pretrained(
#             'openai/clip-vit-base-patch32')

#         self.clip_model.eval()

#         self.clipscore_w = clipscore_w

#         self.image_transform = self._transform(image_size)

#         self.mode = mode
#         assert mode in ['clip_s', 'refclip_s']

#     def _transform(self, n_px):
#         return Compose([
#             Resize(n_px, interpolation=Image.BICUBIC),
#             CenterCrop(n_px),
#             lambda image: image.convert("RGB"),
#             ToTensor(),
#             Normalize((0.48145466, 0.4578275, 0.40821073),
#                       (0.26862954, 0.26130258, 0.27577711)),
#         ])

#     def load_image(self, image_path):
#         image = Image.open(image_path)
#         return image

#     @torch.no_grad()
#     def image_extract(self, image):
#         if isinstance(image, str):
#             image = self.load_image(image)
#         if not isinstance(image, torch.Tensor):
#             image = self.image_transform(image)

#         img_tensor = image.view(-1, 3, 224, 224)
#         device = next(self.clip_model.parameters()).device
#         img_tensor = img_tensor.to(device)

#         clip_model = self.clip_model

#         img_feat = clip_model.vision_model(img_tensor).pooler_output
#         img_feat = clip_model.visual_projection(img_feat)
#         img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

#         return img_feat

#     @torch.no_grad()
#     def text_extract(self, text, prompt="A photo depicts"):
#         if isinstance(text, str):
#             text_batch = [" ".join([prompt, text])]
#         else:
#             text_batch = [" ".join([prompt, txt]) for txt in text]
        
#         input_text = text_batch

#         tokenized = self.tokenizer(
#             input_text, return_tensors='pt', padding=True)

#         input_ids = tokenized.input_ids
#         attention_mask = tokenized.attention_mask

#         clip_model = self.clip_model
#         device = next(self.clip_model.parameters()).device
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)

#         text_feat = clip_model.text_model(input_ids, attention_mask).pooler_output
#         text_feat = clip_model.text_projection(text_feat)
#         text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

#         return text_feat

#     @torch.no_grad()
#     def calc_clip_s(self, img_feat, text_feat):
#         return self.clipscore_w * torch.relu((img_feat * text_feat).sum(dim=-1))

#     @torch.no_grad()
#     def calc_refclip_s(self, img_feat=None, text_feat=None, ref_text_feat=None, ref_text_mask=None, clip_s=None):

#         if clip_s is None:
#             clip_s = self.calc_clip_s(img_feat, text_feat)

#         B, dim = img_feat.size()

#         ref_text_feat = ref_text_feat.view(B, -1, dim)

#         K = ref_text_feat.size(1)

#         text_feat = text_feat.view(B, 1, dim).expand(-1, K, -1)
#         assert ref_text_feat.size() == text_feat.size(), (ref_text_feat.size(), text_feat.size())

#         ref_score = self.calc_clip_s(text_feat, ref_text_feat)
#         if ref_text_mask is not None:
#             if not isinstance(ref_text_mask, torch.Tensor):
#                 ref_text_mask = torch.tensor(ref_text_mask, dtype=ref_score.dtype, device=ref_score.device)
#             ref_score = ref_score.view(B, K) * ref_text_mask.view(B, K)

#         ref_score = ref_score.view(B, K).max(dim=1).values

#         assert clip_s.size() == (B,)
#         assert clip_s.size() == ref_score.size()

#         # harmonic mean
#         refclip_s = 2 / (1 / clip_s + 1 / ref_score)
#         return refclip_s


#     @torch.no_grad()
#     def forward(self,
#                 images=None, text=None,
#                 img_feat=None, text_feat=None,
#                 ref_text=None, ref_text_feat=None, ref_text_mask=None,
#                 prompt="A photo depicts",
#                 mode=None):
#         if img_feat is None:
#             img_feat = self.image_extract(images)
#         img_feat = img_feat.view(-1, 512)

#         if text_feat is None:
#             text_feat = self.text_extract(text, prompt=prompt)
#         text_feat = text_feat.view(-1, 512)

#         if mode is None:
#             mode = self.mode
#         assert mode in ['clip_s', 'refclip_s']

#         if mode == 'clip_s':
#             clip_s = self.calc_clip_s(img_feat, text_feat)
#             return clip_s
#         elif mode == 'refclip_s':
#             if ref_text_feat is None:
#                 ref_text_feat = self.text_extract(ref_text, prompt=prompt)
#             ref_text_feat = ref_text_feat.view(-1, 512)

#             refclip_s = self.calc_refclip_s(img_feat, text_feat, ref_text_feat, ref_text_mask=ref_text_mask)
#             return refclip_s
    
