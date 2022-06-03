import os
import numpy as np
import json
import torch
import torch.nn as nn
import clip
import pytorch_lightning as pl
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from timm.models.vision_transformer import resize_pos_embed
from cog import BasePredictor, Path, Input

import captioning.utils.opts as opts
import captioning.models as models
import captioning.utils.misc as utils


class Predictor(BasePredictor):
    def setup(self):
        import __main__
        __main__.ModelCheckpoint = pl.callbacks.ModelCheckpoint

        self.device = torch.device("cuda:0")
        self.dict_json = json.load(open("./data/cocotalk.json"))
        self.ix_to_word = self.dict_json["ix_to_word"]
        self.vocab_size = len(self.ix_to_word)
        self.clip_model, self.clip_transform = clip.load(
            "RN50", jit=False, device=self.device
        )

        self.preprocess = Compose(
            [
                Resize((448, 448), interpolation=Image.BICUBIC),
                CenterCrop((448, 448)),
                ToTensor(),
            ]
        )

    def predict(
        self,
        image: Path = Input(
            description="Input image.",
        ),
        reward: str = Input(
            choices=["mle", "cider", "clips", "cider_clips", "clips_grammar"],
            default="clips_grammar",
            description="Choose a reward criterion.",
        ),
    ) -> str:

        self.device = torch.device("cuda:0")
        self.dict_json = json.load(open("./data/cocotalk.json"))
        self.ix_to_word = self.dict_json["ix_to_word"]
        self.vocab_size = len(self.ix_to_word)
        self.clip_model, self.clip_transform = clip.load(
            "RN50", jit=False, device=self.device
        )

        self.preprocess = Compose(
            [
                Resize((448, 448), interpolation=Image.BICUBIC),
                CenterCrop((448, 448)),
                ToTensor(),
            ]
        )

        cfg = (
            f"configs/phase1/clipRN50_{reward}.yml"
            if reward == "mle"
            else f"configs/phase2/clipRN50_{reward}.yml"
        )
        print("Loading cfg from", cfg)

        opt = opts.parse_opt(parse=False, cfg=cfg)
        print("vocab size:", self.vocab_size)

        seq_length = 1
        opt.vocab_size = self.vocab_size
        opt.seq_length = seq_length

        opt.batch_size = 1
        opt.vocab = self.ix_to_word
        print(opt.caption_model)

        model = models.setup(opt)
        del opt.vocab

        ckpt_path = opt.checkpoint_path + "-last.ckpt"
        print("Loading checkpoint from", ckpt_path)
        raw_state_dict = torch.load(ckpt_path, map_location=self.device)

        strict = True
        state_dict = raw_state_dict["state_dict"]

        if "_vocab" in state_dict:
            model.vocab = utils.deserialize(state_dict["_vocab"])
            del state_dict["_vocab"]
        elif strict:
            raise KeyError
        if "_opt" in state_dict:
            saved_model_opt = utils.deserialize(state_dict["_opt"])
            del state_dict["_opt"]
            # Make sure the saved opt is compatible with the curren topt
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                if (
                    getattr(saved_model_opt, checkme)
                    in [
                        "updown",
                        "topdown",
                    ]
                    and getattr(opt, checkme) in ["updown", "topdown"]
                ):
                    continue
                assert getattr(saved_model_opt, checkme) == getattr(opt, checkme), (
                    "Command line argument and saved model disagree on '%s' " % checkme
                )
        elif strict:
            raise KeyError
        res = model.load_state_dict(state_dict, strict)
        print(res)

        model = model.to(self.device)
        model.eval()

        image_mean = (
            torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            .to(self.device)
            .reshape(3, 1, 1)
        )
        image_std = (
            torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            .to(self.device)
            .reshape(3, 1, 1)
        )

        num_patches = 196  # 600 * 1000 // 32 // 32
        pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                self.clip_model.visual.attnpool.positional_embedding.shape[-1],
                device=self.device,
            ),
        )
        pos_embed.weight = resize_pos_embed(
            self.clip_model.visual.attnpool.positional_embedding.unsqueeze(0), pos_embed
        )
        self.clip_model.visual.attnpool.positional_embedding = pos_embed

        with torch.no_grad():
            image = self.preprocess(Image.open(str(image)).convert("RGB"))
            image = torch.tensor(np.stack([image])).to(self.device)
            image -= image_mean
            image /= image_std

            tmp_att, tmp_fc = self.clip_model.encode_image(image)
            tmp_att = tmp_att[0].permute(1, 2, 0)

            att_feat = tmp_att

        # Inference configurations
        eval_kwargs = {}
        eval_kwargs.update(vars(opt))

        with torch.no_grad():
            fc_feats = torch.zeros((1, 0)).to(self.device)
            att_feats = att_feat.view(1, 196, 2048).float().to(self.device)
            att_masks = None

            # forward the model to also get generated samples for each image
            # Only leave one feature for each image, in case duplicate sample
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({"sample_n": 1})
            seq, seq_logprobs = model(
                fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode="sample"
            )
            seq = seq.data

            sents = utils.decode_sequence(model.vocab, seq)

        return sents[0]
