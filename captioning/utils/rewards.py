from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from collections import OrderedDict
import torch

import sys
try:
    sys.path.append("cider")
    from pyciderevalcap.ciderD.ciderD import CiderD
    from pyciderevalcap.cider.cider import Cider
    sys.path.append("coco-caption")
    from pycocoevalcap.bleu.bleu import Bleu
except:
    print('cider or coco-caption missing')

CiderD_scorer = None
Cider_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')


from .misc import decode_sequence

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Cider_scorer
    Cider_scorer = Cider_scorer or Cider(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result, opt):
    batch_size = len(data_gts) 
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts_, res_)
        if hasattr(opt, 'verbose') and not opt.verbose:
            pass
        else:
            print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts_, res__)
        bleu_scores = np.array(bleu_scores[3])
        if hasattr(opt, 'verbose') and not opt.verbose:
            pass
        else:
            print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    unnormalized_reward_mean = scores[:gen_result_size].flatten().mean()

    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]

    scores = scores.reshape(gen_result_size)

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards, unnormalized_reward_mean


def get_self_critical_clipscore_reward(greedy_res, data_gts, gen_result, opt, clipscore_model, clip_vis_feats, vocab):
    batch_size = len(data_gts) 
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size

    B = batch_size
    K = seq_per_img
    L = gen_result.shape[1]
    assert gen_result.shape == (B*K , L)

    # res = OrderedDict()
    # gen_result = gen_result.data.cpu().numpy()
    # greedy_res = greedy_res.data.cpu().numpy()
    # for i in range(gen_result_size):
    #     res[i] = [array_to_str(gen_result[i])]
    # for i in range(batch_size):
    #     res[gen_result_size + i] = [array_to_str(greedy_res[i])]

    # gts = OrderedDict()
    # for i in range(len(data_gts)):
    #     gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    # res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    # res__ = {i: res[i] for i in range(len(res_))}
    # gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    # gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})

    # res = []
    # gen_result = gen_result.data.cpu().numpy()
    # greedy_res = greedy_res.data.cpu().numpy()
    # # for i in range(gen_result_size):
    #     # res.append(array_to_str(gen_result[i]))
    # res.extend(decode_sequence(vocab, gen_result))
        
        
    # # for i in range(batch_size):
    # #     res.append(array_to_str(greedy_res[i]))
    # res.extend(decode_sequence(vocab, greedy_res))

    if clipscore_model.mode == 'refclip_s':
        gts = []
        gts_valid_mask = []
        max_n_refs = max([len(_gts) for _gts in data_gts])
        for i in range(len(data_gts)):
            _gts = decode_sequence(vocab, data_gts[i])
            # pad references
            n_ref = len(_gts)
            _gts.extend([''] * (max_n_refs - n_ref))
            gts.extend(_gts)
            gts_valid_mask.extend([1] * n_ref + [0] * (max_n_refs - n_ref))
        assert len(gts) == B * max_n_refs
        assert len(gts_valid_mask) == B * max_n_refs

        # print(gts)
        # print(gts_valid_mask)
        # exit()
        

    # assert len(res) == B * K + B, len(res)

    # print(res)
    # exit()
    
    if opt.clipscore_reward_weight > 0:
        with torch.no_grad():
            clipscore_model.eval()

            # 1) calculate reward
            gen_result = gen_result.data.cpu().numpy()
            res = decode_sequence(vocab, gen_result)
            assert len(res) == B * K, len(res)

            # [B * K, dim)
            if getattr(opt, 'use_grammar', False) and not getattr(opt, 'joint_out', False):
                text_pre_feat = clipscore_model.text_extract(res, proj_norm=False)
                
                grammar_logit = clipscore_model.grammar_score_head(text_pre_feat.view(-1, 512))
                grammar_prob = torch.softmax(grammar_logit, dim=-1)[:, 1]
                grammar_prob = grammar_prob.view(B*K).detach()

                text_feat = clipscore_model.clip_model.text_projection(text_pre_feat)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            else:
                text_feat = clipscore_model.text_extract(res)


            assert text_feat.size() == (B * K, 512), text_feat.size()
            assert clip_vis_feats.size() == (B, 512), clip_vis_feats.size()

            # [B * K, dim]
            vis_feat = clip_vis_feats.view(B, 1, -1).expand(-1, K, -1).contiguous().view(B * K, -1)

            clip_s = clipscore_model(text_feat=text_feat, img_feat=vis_feat, mode='clip_s')
            clip_s = clip_s.view(B * K).detach()

            if clipscore_model.mode == 'refclip_s':
                # [B * n_ref, dim]
                ref_text_feat = clipscore_model.text_extract(gts)
                ref_text_mask = torch.tensor(gts_valid_mask, dtype=ref_text_feat.dtype, device=ref_text_feat.device)

                assert ref_text_feat.size() == (B * max_n_refs, 512), ref_text_feat.size()
                assert ref_text_mask.size() == (B * max_n_refs,), ref_text_mask.size()

                # [B * K]
                refclip_s = clipscore_model.calc_refclip_s(
                    text_feat=text_feat, img_feat=vis_feat,
                    ref_text_feat=ref_text_feat.view(B, 1, max_n_refs, -1).expand(-1, K, -1, -1).contiguous().view(B * K * max_n_refs, -1),
                    ref_text_mask=ref_text_mask.view(B, 1, max_n_refs).expand(-1, K, -1).contiguous().view(B * K * max_n_refs),
                    clip_s=clip_s)
                refclip_s = refclip_s.view(B * K).detach()

            # 2) calcualte reward for baseline (greedy)
            greedy_res = greedy_res.data.cpu().numpy()
            res = decode_sequence(vocab, greedy_res)
            assert len(res) == B, len(res)

            # [B, dim)

            if getattr(opt, 'use_grammar', False) and getattr(opt, 'use_grammar_baseline', False) and not getattr(opt, 'joint_out', False):
                text_pre_feat = clipscore_model.text_extract(res, proj_norm=False)
                
                grammar_logit = clipscore_model.grammar_score_head(text_pre_feat.view(-1, 512))
                grammar_prob_baseline = torch.softmax(grammar_logit, dim=-1)[:, 1]
                grammar_prob_baseline = grammar_prob_baseline.view(B).detach()

                text_feat = clipscore_model.clip_model.text_projection(text_pre_feat)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            else:
                text_feat = clipscore_model.text_extract(res)

            assert text_feat.size() == (B, 512), text_feat.size()
            assert clip_vis_feats.size() == (B, 512), clip_vis_feats.size()

            vis_feat = clip_vis_feats.view(B, 512)

            # [B]
            clip_s_baseline = clipscore_model(text_feat=text_feat, img_feat=vis_feat, mode='clip_s')
            clip_s_baseline = clip_s_baseline.view(B).detach()

            if clipscore_model.mode == 'refclip_s':
                # # [B * n_ref]
                # ref_text_feat = clipscore_model.text_extract(gts)
                # ref_text_mask = torch.tensor(gts_valid_mask, dtype=ref_text_feat.dtype, device=ref_text_feat.device)
                # assert ref_text_feat.size() == (B * max_n_refs, 512), ref_text_feat.size()
                # assert ref_text_mask.size() == (B * max_n_refs), ref_text_mask.size()

                # [B]
                refclip_s_baseline = clipscore_model.calc_refclip_s(
                    text_feat=text_feat, img_feat=vis_feat,
                    ref_text_feat=ref_text_feat,
                    ref_text_mask=ref_text_mask,
                    clip_s=clip_s_baseline)
                refclip_s_baseline = refclip_s_baseline.view(B).detach()

            if clipscore_model.mode == 'clip_s':
                rewards = clip_s - clip_s_baseline.view(B, 1).expand(-1, K).contiguous().flatten()
                unnormalized_mean_reward = clip_s.mean()
            elif clipscore_model.mode == 'refclip_s':
                rewards = refclip_s - refclip_s_baseline.view(B, 1).expand(-1, K).contiguous().flatten()
                unnormalized_mean_reward = refclip_s.mean()
            
            # # [B * K + B, dim)
            # text_feat = clipscore_model.text_extract(res)
            # assert text_feat.size() == (B * K + B, 512), text_feat.size()

            # assert clip_vis_feats.size() == (B, 512), clip_vis_feats.size()

            # # [B, dim] -> [B * K + B, dim]
            # # vis_feat = clip_vis_feats.view(B, 1, -1).expand(-1, K + 1, -1).contiguous().view(B * (K + 1), -1)
            # # vis_feat = clip_vis_feats.view(1, B, -1).expand(K + 1, -1, -1).contiguous().view((K + 1) * B, -1)

            # # [B * K, dim]
            # gen_vis_feat = clip_vis_feats.view(B, 1, -1).expand(-1, K, -1).contiguous().view(B * K, -1)
            # # [B, dim]
            # greedy_vis_feat = clip_vis_feats
            # # [B * K + B, dim]
            # vis_feat = torch.cat([gen_vis_feat, greedy_vis_feat], dim=0)

            # # if clipscore_model.mode == 'clip_s':
            # # [B * K + B, dim]
            # clip_s = clipscore_model(text_feat=text_feat, img_feat=vis_feat)
            # clip_s = clip_s.view(B * K + B).detach()
            
            
            # if clipscore_model.mode == 'refclip_s':
            #     # [B * K, dim]
            #     ref_text_feat = clipscore_model.text_extract(gts)

            #     clipscore_scores = clipscore_model.calc_refclip_s(text_feat=text_feat, img_feat=vis_feat, ref_text_feat=ref_text_feat, clip_s=clip_s)
            #     clipscore_scores = clipscore_scores.view(B * K + B).detach()

            if getattr(opt, 'use_grammar', False) and not getattr(opt, 'joint_out', False):
            
                if getattr(opt, 'use_grammar_baseline', False):
                    grammar_rewards = grammar_prob - grammar_prob_baseline.view(B, 1).expand(-1, K).contiguous().flatten()
                else:
                    grammar_rewards = grammar_prob
            else:
                grammar_rewards = None


        if hasattr(opt, 'verbose') and not opt.verbose:
            pass
        else:
            if clipscore_model.mode == 'clip_s':
                print('CLIP-S:', rewards)
            elif clipscore_model.mode == 'refclip_s':
                print('RefCLIP-S:', rewards)
    else:
        rewards = torch.zeros(B, L)
        unnormalized_mean_reward = None
        grammar_rewards = None
            

    rewards = opt.clipscore_reward_weight * rewards


    # scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    # scores = scores.reshape(gen_result_size)
    # rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    # [B, K]
    # scores = scores[:gen_result_size].reshape(B, K) - scores[-B:].unsqueeze(1)

    # [B*K, L]
    # rewards = scores.view(-1, 1).expand(-1, L).contiguous()
    rewards = rewards.view(-1, 1).expand(-1, L).contiguous()

    if getattr(opt, 'use_grammar', False) and not getattr(opt, 'joint_out', False):
        grammar_rewards = grammar_rewards.view(-1, 1).expand(-1, L).contiguous()

    return rewards, unnormalized_mean_reward, grammar_rewards

def get_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        # print('Cider scores:', _)
        if hasattr(opt, 'verbose') and not opt.verbose:
            pass
        else:
            print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        # print('Bleu scores:', _[3])
        if hasattr(opt, 'verbose') and not opt.verbose:
            pass
        else:
            print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    return scores

def get_self_cider_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = []
    
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res.append(array_to_str(gen_result[i]))

    scores = []
    for i in range(len(data_gts)):
        tmp = Cider_scorer.my_self_cider([res[i*seq_per_img:(i+1)*seq_per_img]])
        def get_div(eigvals):
            eigvals = np.clip(eigvals, 0, None)
            return -np.log(np.sqrt(eigvals[-1]) / (np.sqrt(eigvals).sum())) / np.log(len(eigvals))
        scores.append(get_div(np.linalg.eigvalsh(tmp[0]/10)))

    scores = np.array(scores)

    return scores
