import torch
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward, get_self_critical_clipscore_reward
from ..utils.clipscore import CLIPScore
import numpy as np

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)

        self.clipscore_model = None
        if self.opt.use_clipscore:
            use_grammar = getattr(self.opt, 'use_grammar', False)
            joint_out = getattr(self.opt, 'joint_out', False)
            self.clipscore_model = CLIPScore(
                mode=opt.clipscore_mode,
                use_grammar=use_grammar,
                joint_out=joint_out,
                )
            for p in self.clipscore_model.parameters():
                p.requires_grad = False

            if use_grammar:
                state_dict = torch.load(self.opt.clip_load_path, map_location='cpu')
                self.clipscore_model.load_state_dict(state_dict['state_dict'])

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, clip_vis_feats=None):
        opt = self.opt
        
        out = {}
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]

            if getattr(self.opt, 'use_multi_rewards', False):
                assert self.opt.use_clipscore
                clipscore_reward_normalized, clipscore_unnormalized_mean, grammar_rewards = get_self_critical_clipscore_reward(
                    greedy_res, gts, gen_result, self.opt, self.clipscore_model, clip_vis_feats, self.model.vocab)

                if self.opt.clipscore_mode == 'clip_s':
                    out['CLIP-S'] = clipscore_unnormalized_mean
                elif self.opt.clipscore_mode == 'refclip_s':
                    out['RefCLIP-S'] = clipscore_unnormalized_mean

                if getattr(self.opt, 'use_grammar', False):
                    out['grammar_reward'] = grammar_rewards.mean()

                    reward = clipscore_reward_normalized + grammar_rewards


                else:
                    assert grammar_rewards is None

                    cider_reward_normalized, cider_unnormalized_mean = get_self_critical_reward(
                        greedy_res, gts, gen_result, self.opt)
                    out['CIDEr'] = cider_unnormalized_mean
                    if isinstance(cider_reward_normalized, np.ndarray):
                        cider_reward_normalized = torch.from_numpy(cider_reward_normalized).to(clipscore_reward_normalized.device)

                    reward = clipscore_reward_normalized + cider_reward_normalized
            else:
                if self.opt.use_clipscore:
                    clipscore_reward_normalized, clipscore_unnormalized_mean, _ = get_self_critical_clipscore_reward(
                        greedy_res, gts, gen_result, self.opt, self.clipscore_model, clip_vis_feats, self.model.vocab)
                    if self.opt.clipscore_mode == 'clip_s':
                        out['CLIP-S'] = clipscore_unnormalized_mean
                    elif self.opt.clipscore_mode == 'refclip_s':
                        out['RefCLIP-S'] = clipscore_unnormalized_mean
                    reward = clipscore_reward_normalized
                else:
                    cider_reward_normalized, cider_unnormalized_mean = get_self_critical_reward(
                        greedy_res, gts, gen_result, self.opt)
                    out['CIDEr'] = cider_unnormalized_mean
                    reward = cider_reward_normalized

            if isinstance(reward, np.ndarray):
                reward = torch.from_numpy(reward)
            reward = reward.to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out

