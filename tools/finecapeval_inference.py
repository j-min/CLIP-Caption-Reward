import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import time
import os
from collections import defaultdict
import json

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.pth_loader import CaptionDataset
import captioning.utils.eval_utils as eval_utils
# import captioning.utils.vizwiz_eval_utils as vizwiz_eval_utils
import captioning.utils.misc as utils
from captioning.utils.rewards import init_scorer, get_self_critical_reward
from captioning.modules.loss_wrapper import LossWrapper

import pytorch_lightning as pl


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_keyboard_interrupt(self, trainer, pl_module):
        # Save model when keyboard interrupt
        filepath = os.path.join(self.dirpath, self.prefix + 'interrupt.ckpt')
        self._save_model(filepath)


if __name__ == '__main__':

    device = 'cuda'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward', type=str, default='mle')
    args = parser.parse_args()

    if args.reward == 'mle':
        cfg = f'configs/phrase1/fg_clipRN50_{args.reward}.yml'
    else:
        cfg = f'configs/phrase2/fg_clipRN50_{args.reward}.yml'
    # ckpt_fname = f'{args.reward}-last.ckpt    

    # if args.model_name == 'mle':
    #     cfg = 'configs/phrase2/clipRN50_mle.yml'
    #     ckpt_fname = 'mle-last.ckpt'

    # elif args.model_name == 'clips':
    #     cfg = 'configs/phrase2/clipRN50_clips.yml'

    # elif args.model_name == 'cider':

    # elif args.model_name == 'clips_cider':

    # elif args.model_name == 'clips_grammar':

    # # # CLIP-S
    # model_name = "clips"
    # cfg = 'configs/phrase2/clipRN50_clips.yml'
    # ckpt_fname = 'clipscore-last.ckpt'

    # # CIDER
    # model_name = "cider"
    # cfg = 'configs/phrase2/clipRN50_cider.yml'
    # ckpt_fname = 'cider-last.ckpt'

    # # CLIP-S + CIDER
    # model_name = "cider_clips"
    # cfg = 'configs/phrase2/clipRN50_cider_clips.yml'
    # ckpt_fname = 'cider_clipslast.ckpt'

    # CLIP-S + Grammar
    # model_name = "clips_grammar"
    # cfg = 'configs/phrase2/clipRN50_clips_grammar.yml'

    # ckpt_fname = f'{model_name}-last.ckpt'

    print("Loading cfg from", cfg)

    opt = opts.parse_opt(parse=False, cfg=cfg)

    dataset = CaptionDataset(opt)

    opt.vocab_size = dataset.vocab_size
    opt.seq_length = dataset.seq_length

    opt.batch_size = 40

    opt.vocab = dataset.get_vocab()

    model = models.setup(opt)
    del opt.vocab

    ckpt_path = opt.checkpoint_path + '-last.ckpt'

    # /print("Loading checkpoint from", opt.checkpoint_path+, ckpt_fname))
    print("Loading checkpoint from", ckpt_path)
    raw_state_dict = torch.load(
        ckpt_path,
        map_location=device)

    strict = True

    state_dict = raw_state_dict['state_dict']

    if '_vocab' in state_dict:
        model.vocab = utils.deserialize(state_dict['_vocab'])
        del state_dict['_vocab']
    elif strict:
        raise KeyError
    if '_opt' in state_dict:
        saved_model_opt = utils.deserialize(state_dict['_opt'])
        del state_dict['_opt']
        # Make sure the saved opt is compatible with the curren topt
        need_be_same = ["caption_model",
                        "rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            if getattr(saved_model_opt, checkme) in ['updown', 'topdown'] and \
                    getattr(opt, checkme) in ['updown', 'topdown']:
                continue
            assert getattr(saved_model_opt, checkme) == getattr(
                opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
    elif strict:
        raise KeyError
    res = model.load_state_dict(state_dict, strict)
    print(res)

    opt.use_grammar = False

    lw_model = LossWrapper(model, opt)

    split = 'test'

    print("Building dataloader...")

    test_dataset = torch.utils.data.Subset(
        dataset,
        dataset.split_ix[split]
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=dataset.collate_func
    )

    # test_iter = iter(test_loader)
    # batch = next(test_iter)

    eval_kwargs = {'dataset': opt.input_json}
    eval_kwargs.update(vars(opt))

    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    # num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    # lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)

    crit = lw_model.crit

    model = model.to(device)

    from tqdm import tqdm

    test_id2sent = {}

    model.eval()

    print("running inference...")

    for data in tqdm(test_loader):
        with torch.no_grad():
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'],
                data['labels'], data['masks'], data['att_masks']]
            tmp = [d.to(device) if isinstance(d, torch.Tensor) else d for d in tmp]

            fc_feats, att_feats, labels, masks, att_masks = tmp

            loss = crit(model(fc_feats, att_feats,
                            labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])

            # forward the model to also get generated samples for each image
            # Only leave one feature for each image, in case duplicate sample
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1})
            seq, seq_logprobs = model(
                fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
            seq = seq.data
            entropy = - (F.softmax(seq_logprobs, dim=2) *
                        seq_logprobs).sum(2).sum(1) / ((seq > 0).to(seq_logprobs).sum(1)+1)
            perplexity = - \
                seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(
                    2).sum(1) / ((seq > 0).to(seq_logprobs).sum(1)+1)

            # Print beam search
            if beam_size > 1 and verbose_beam:
                for i in range(fc_feats.shape[0]):
                    print('\n'.join([utils.decode_sequence(model.vocab, _[
                        'seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                    print('--' * 10)
            sents = utils.decode_sequence(model.vocab, seq)

        for d, sent in zip(data['infos'], sents):
            test_id2sent[d['id']] = sent

    res_path = f'FineCapEval_results/clipRN50_{args.reward}.json'

    print("Results save at {}".format(res_path))

    with open(res_path, 'w') as f:
        json.dump(test_id2sent, f)


