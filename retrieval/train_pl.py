from ast import parse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import time
import os
from collections import defaultdict

# import captioning.utils.opts as opts
# import captioning.models as models
# from captioning.data.pth_loader import CaptionDataset
# import captioning.utils.eval_utils as eval_utils
# import captioning.utils.misc as utils
# from captioning.utils.rewards import init_scorer, get_self_critical_reward
# from captioning.modules.loss_wrapper import LossWrapper

from clip_model import CLIPScore
from caption_data import COCORetrievalDataset

import pytorch_lightning as pl

import detectron2.utils.comm as d2comm
from detectron2.utils.env import seed_all_rng
seed_all_rng(1234)


class LitModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.args = args
        # Intilaize dataset
        # self.dataset = CaptionDataset(opt)

        # self.dataset = 

        # opt.vocab_size = self.dataset.vocab_size
        # opt.seq_length = self.dataset.seq_length
        # self.batch_size = opt.batch_size

        # Build model
        # opt.vocab = self.dataset.get_vocab()
        # model = models.setup(opt)
        # print(model)
        # del opt.vocab

        # wrapper with loss in it.
        # lw_model = LossWrapper(model, opt)

        self.model = CLIPScore(use_grammar=opt.use_grammar, joint_out=opt.joint_out)
        # self.lw_model = lw_model

        for p in self.model.clip_model.vision_model.parameters():
            p.requires_grad = False
        for p in self.model.clip_model.visual_projection.parameters():
            p.requires_grad = False

        # self.struc_flag = None
        # self.sc_flag = None


    def forward(self, *args, **kwargs):
        """
        I hate this design. Never pretend it as a nn.Module
        """
        raise NotImplementedError

    def train_dataloader(self):
        # train_dataset = torch.utils.data.Subset(
        #     self.dataset,
        #     self.dataset.split_ix['train']
        # )

        # train_loader = torch.utils.data.DataLoader(
        #     dataset=train_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=4,
        #     collate_fn=self.dataset.collate_func
        # )

        train_dataset = COCORetrievalDataset(
            split='karpathy_train', mode='train',
            args=opt,
            verbose=verbose
            )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=train_dataset.collate_fn
        )

        return train_loader

    def val_dataloader(self, split='karpathy_val'):
        # val_dataset = torch.utils.data.Subset(
        #     self.dataset,
        #     self.dataset.split_ix[split]
        # )
        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=4,
        #     drop_last=False,
        #     collate_fn=self.dataset.collate_func
        # )

        val_dataset = COCORetrievalDataset(
            split=split, mode='val',
            args=opt,
            verbose=verbose
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=opt.valid_batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            collate_fn=val_dataset.collate_fn
        )

        return val_loader

    def test_dataloader(self):

        return self.val_dataloader('karpathy_test')

    def training_step(self, data, batch_idx):


        batch = data
        self.model.train()

        model_out = self.model.train_step(
            img_feat=batch['img_feats'],
            text=batch['text'],
            neg_text=batch['neg_text'],
        )

        clip_loss = model_out['clip_loss']

        if self.opt.joint_out:
            loss = clip_loss
        else:
            grammar_loss = model_out['grammar_loss']
            loss = clip_loss + grammar_loss
            

        data_time = self.trainer.profiler.recorded_durations["get_train_batch"][-1]
        data_time = torch.tensor(data_time)

        # print('batch_idx', batch_idx)
        # print('loss:', loss)

        # logger_logs = model_out.copy()
        logger_logs = {}
        
        logger_logs['loss'] = loss.detach()

        logger_logs['clip_loss'] = clip_loss.detach()

        if not self.opt.joint_out:
            logger_logs['grammar_loss'] = grammar_loss.detach()

        logger_logs['data_time'] = data_time.detach()

        # UserWarning: The {progress_bar:dict keyword} was deprecated in 0.9.1 and will be removed in 1.0.0
        # Please use self.log(...) inside the lightningModule instead.

        # # log on a step or aggregate epoch metric to the logger and/or progress bar
        # # (inside LightningModule)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # warnings.warn(*args, **kwargs)
        # UserWarning: The {log:dict keyword} was deprecated in 0.9.1 and will be removed in 1.0.0
        # Please use self.log(...) inside the lightningModule instead.

        # output = {
        #     'loss': loss,
        #     'log': logger_logs,
        #     'progress_bar': {'data_time': data_time}
        # }

        for k, v in logger_logs.items():
            if k in ['data_time', 'clip_loss', 'grammar_loss']:
                self.log('train/'+k, v, prog_bar=True)
            else:
                self.log('train/'+k, v)
        
        # print('training step logged')

        return loss

    def validation_step(self, data, batch_idx):

        batch = data
        self.model.eval()

        with torch.no_grad():
            model_out = self.model.train_step(
                img_feat=batch['img_feats'],
                text=batch['text'],
                neg_text=batch['neg_text'],
            )

            if self.opt.joint_out:
                clip_loss = model_out['clip_loss']
                loss = clip_loss

                output = {
                    # 'val_loss': loss,
                    'loss': loss.detach(),
                    'clip_loss': clip_loss.detach(),
                    # 'grammar_loss': grammar_loss.detach(),

                    'img_feat': model_out['img_feat'].detach(),
                    'text_feat': model_out['text_feat'].detach(),
                    # 'neg_text_feat': model_out['neg_text_feat'].detach(),
                    # 'grammar_pos_pred': model_out['grammar_pos_pred'].detach(),
                    # 'grammar_neg_pred': model_out['grammar_neg_pred'].detach(),
                    # 'predictions': predictions,
                    # 'n_predictions': n_predictions,
                }
            else:
                clip_loss = model_out['clip_loss']
                grammar_loss = model_out['grammar_loss']
                loss = clip_loss + grammar_loss

                output = {
                    # 'val_loss': loss,
                    'loss': loss.detach(),
                    'clip_loss': clip_loss.detach(),
                    'grammar_loss': grammar_loss.detach(),

                    'img_feat': model_out['img_feat'].detach(),
                    'text_feat': model_out['text_feat'].detach(),
                    # 'neg_text_feat': model_out['neg_text_feat'].detach(),
                    'grammar_pos_pred': model_out['grammar_pos_pred'].detach(),
                    'grammar_neg_pred': model_out['grammar_neg_pred'].detach(),
                    # 'predictions': predictions,
                    # 'n_predictions': n_predictions,
                }
        return output

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def validation_epoch_end(self, outputs, split='val'):
        outputs = d2comm.gather(outputs)
        # master node
        if d2comm.is_main_process():
            assert self.trainer.node_rank == 0 and self.trainer.local_rank == 0
            outputs = sum(outputs, [])

            out = {}

            val_loss_mean = sum([_['loss'].cpu() for _ in outputs]) / len(outputs)
            val_clip_loss_mean = sum([_['clip_loss'].cpu() for _ in outputs]) / len(outputs)
            if not self.opt.joint_out:
                val_grammar_loss_mean = sum([_['grammar_loss'].cpu() for _ in outputs]) / len(outputs)

            print('loss', val_loss_mean.item())
            print('clip_loss', val_clip_loss_mean.item())
            if not self.opt.joint_out:
                print('grammar_loss', val_grammar_loss_mean.item())

            logit_scale = self.model.clip_model.logit_scale.exp().cpu()

            text_feats = torch.cat([_['text_feat'].cpu() for _ in outputs], dim=0)
            img_feats = torch.cat([_['img_feat'].cpu() for _ in outputs], dim=0)

            assert text_feats.size() == (5000, 512), text_feats.size()
            assert img_feats.size() == (5000, 512), img_feats.size()

            logits_per_text = torch.matmul(text_feats, img_feats.t()) * logit_scale
            logits_per_image = logits_per_text.T

            # text-to-image retrieval
            print('Text-to-Image retrieval')
            for k in [1, 5, 10]:
                text_to_image_topk = logits_per_text.topk(k, dim=1).indices

                n_text = len(text_to_image_topk)

                labels = torch.arange(0, n_text).view(-1, 1)

                n_retrieved = ((text_to_image_topk == labels).sum(dim=1) > 0).sum()

                recall_k = n_retrieved / n_text * 100

                out[f'text_to_image_recall_{k}'] = recall_k.item()

                print(f'R@{k}: {recall_k.item():.2f}%')

            # image-to-text retrieval
            print('Image-to-Text retrieval')
            for k in [1, 5, 10]:
                image_to_text_topk = logits_per_image.topk(k, dim=1).indices

                n_image = len(image_to_text_topk)

                labels = torch.arange(0, n_image).view(-1, 1)

                n_retrieved = ((image_to_text_topk == labels).sum(dim=1) > 0).sum()

                recall_k = n_retrieved / n_image * 100

                out[f'image_to_text_recall_{k}'] = recall_k.item()

                print(f'R@{k}: {recall_k.item():.2f}%')

            out.update({
                'loss': val_loss_mean.item(),
                'clip_loss': val_clip_loss_mean.item()
            })

            if not self.opt.joint_out:
                # grammar scoring
                grammar_pos_pred = torch.cat([_['grammar_pos_pred'].cpu() for _ in outputs], dim=0)
                grammar_neg_pred = torch.cat([_['grammar_neg_pred'].cpu() for _ in outputs], dim=0)

                TP = (grammar_pos_pred == 1).sum().item()
                FP = (grammar_pos_pred == 0).sum().item()
                FN = (grammar_neg_pred == 1).sum().item()
                TN = (grammar_neg_pred == 0).sum().item()
                print('Grammar check')
                print(f'TP: {TP} FP: {FP}  FN: {FN}  TN: {TN}')

                precision = TP / (TP + FP) * 100
                recall = TP / (TP + FN) * 100
                accuracy = (TP + TN) / (TP + FP + FN + TN) * 100
                f1 = 2 * precision * recall / (precision + recall)
                print(f'Precision: {precision:.2f}%')
                print(f'Recall: {recall:.2f}%')
                print(f'Accuracy: {accuracy:.2f}%')
                print(f'F1: {f1:.2f}%')
                print('Total: {}'.format(len(grammar_pos_pred)))

                out.update({
                    'grammar_loss': val_grammar_loss_mean,

                    'grammar_precision': precision,
                    'grammar_recall': recall,
                    'grammar_accuracy': accuracy,
                    'grammar_f1': f1,

                })

        else:
            out = {}

        out = d2comm.all_gather(out)[0]  # Only the one from master node
        assert len(out) > 0  # make sure the head has index 0

        # must all be tensors
        out = {k: torch.tensor(v) if not torch.is_tensor(
            v) else v for k, v in out.items()}

        for k, v in out.items():
            self.log(f'{split}/{k}', v)

    def test_epoch_end(self, outputs):

        self.validation_epoch_end(outputs, 'test')
        
    def configure_optimizers(self):
        # opt = self.opt
        # model = self.model

        # parameters = [p for p in model.parameters() if p.requires_grad]

        # if opt.noamopt:
        #     # assert opt.caption_model in ['transformer', 'bert', 'm2transformer'], 'noamopt can only work with transformer'
        #     optimizer = utils.get_std_opt(
        #         model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        # elif opt.reduce_on_plateau:
        #     # optimizer = utils.build_optimizer(model.parameters(), opt)
        #     optimizer = utils.build_optimizer(parameters, opt)
        #     optimizer = utils.ReduceLROnPlateau(optimizer,
        #                                         factor=opt.reduce_on_plateau_factor,
        #                                         patience=opt.reduce_on_plateau_patience)
        # else:
        #     # optimizer = utils.build_optimizer(model.parameters(), opt)
        #     optimizer = utils.build_optimizer(parameters, opt)


        # from transformers.optimization import AdamW, get_linear_schedule_with_warmup
        # batch_per_epoch = len(self.train_loader)
        # t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
        # warmup_ratio = self.args.warmup_ratio
        # warmup_iters = int(t_total * warmup_ratio)
        # if self.verbose:
        #     print("Batch per epoch: %d" % batch_per_epoch)
        #     print("Total Iters: %d" % t_total)
        #     print('Warmup ratio:', warmup_ratio)
        #     print("Warm up Iters: %d" % warmup_iters)

        if self.args.optim == 'adamw':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            for group in optimizer_grouped_parameters:
                group['params'] = [p for p in group['params'] if p.requires_grad]

            from transformers.optimization import AdamW
            optim = AdamW(optimizer_grouped_parameters,
                            lr=self.args.lr, eps=self.args.adam_eps)
            # lr_scheduler = get_linear_schedule_with_warmup(
            #     optim, warmup_iters, t_total)

        # optimizers = []
        optimizers = [optim]
        lr_schedulers = []

        return optimizers, lr_schedulers

    def optimizer_step(self, epoch, batch_idx, optimizer,
                       optimizer_idx, *args, **kwargs):
        # # warm up lr
        # opt = self.opt
        # iteration = self.trainer.global_step
        # if opt.use_warmup and (iteration < opt.noamopt_warmup):
        #     opt.current_lr = opt.learning_rate * \
        #         (iteration+1) / opt.noamopt_warmup
        #     utils.set_lr(optimizer, opt.current_lr)

        super().optimizer_step(epoch, batch_idx, optimizer,
                               optimizer_idx, *args, **kwargs)

        # print('optimizer step')

    def state_dict(self):
        """
        Save the model state dict as well as opt and vocab
        """
        state_dict = self.model.state_dict()
        device = next(iter(state_dict.values())).device
        assert '_vocab' not in state_dict and '_opt' not in state_dict, 'Just in case'
        # state_dict.update({
        #     '_vocab': utils.serialize_to_tensor(self.model.vocab).to(device),
        #     '_opt': utils.serialize_to_tensor(self.opt).to(device)
        # })
        return state_dict

    def load_state_dict(self, state_dict=None, strict=True):
        # if '_vocab' in state_dict:
        #     self.model.vocab = utils.deserialize(state_dict['_vocab'])
        #     del state_dict['_vocab']
        # elif strict:
        #     raise KeyError
        # if '_opt' in state_dict:
        #     saved_model_opt = utils.deserialize(state_dict['_opt'])
        #     del state_dict['_opt']
        #     opt = self.opt
        #     # Make sure the saved opt is compatible with the curren topt
        #     need_be_same = ["caption_model",
        #                     "rnn_type", "rnn_size", "num_layers"]
        #     for checkme in need_be_same:
        #         if getattr(saved_model_opt, checkme) in ['updown', 'topdown'] and \
        #                 getattr(opt, checkme) in ['updown', 'topdown']:
        #             continue
        #         assert getattr(saved_model_opt, checkme) == getattr(
        #             opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
        # elif strict:
        #     raise KeyError
        self.model.load_state_dict(state_dict, strict)


class OnEpochStartCallback(pl.Callback):

    def on_epoch_start(self, trainer, pl_module):
        # Update lr/training stage/scheduled sampling prob etc.
        opt = pl_module.opt
        model = pl_module.model
        epoch = trainer.current_epoch
        optimizer = trainer.optimizers[0]

        # if not opt.noamopt and not opt.reduce_on_plateau:
        #     # Assign the learning rate
        #     if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
        #         frac = (
        #             epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
        #         decay_factor = opt.learning_rate_decay_rate ** frac
        #         opt.current_lr = opt.learning_rate * decay_factor
        #     else:
        #         opt.current_lr = opt.learning_rate
        #     utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
        # # Assign the scheduled sampling prob
        # if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
        #     frac = (
        #         epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
        #     opt.ss_prob = min(opt.scheduled_sampling_increase_prob *
        #                       frac, opt.scheduled_sampling_max_prob)
        #     model.ss_prob = opt.ss_prob

        # # If start self critical training
        # if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
        #     sc_flag = True
        #     init_scorer(opt.cached_tokens)
        # else:
        #     sc_flag = False

        # # If start structure loss training
        # if opt.structure_after != -1 and epoch >= opt.structure_after:
        #     struc_flag = True
        #     init_scorer(opt.cached_tokens)
        # else:
        #     struc_flag = False

        # pl_module.struc_flag = struc_flag
        # pl_module.sc_flag = sc_flag


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_keyboard_interrupt(self, trainer, pl_module):
        # Save model when keyboard interrupt
        filepath = os.path.join(self.dirpath, self.prefix + 'interrupt.ckpt')
        self._save_model(filepath)

from param import parse_args
# opt = opts.parse_opt()
args = parse_args()
opt = args

checkpoint_callback = ModelCheckpoint(
    filepath=opt.checkpoint_dir + '{epoch:02d}',
    # dirpath=opt.checkpoint_path,
    save_last=True,
    save_top_k=1,
    verbose=True,
    # monitor='to_monitor',
    # monitor='val/to_monitor',
    # monitor='val/CIDEr',
    monitor='val/loss',
    mode='min',
    # prefix=opt.id+'_',
    prefix=opt.id,
    # filename=f'{opt.id}_',
)

verbose = True
# import torch
# if torch.cuda.current_device() in [0, -1]:
if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK'] != '0':
    verbose = False

# if verbose:
#     print(opt)
#     print("""
#     val_image_use,
#     save_checkpoint_very
#     save_every_epoch,
#     save_history-ckpt will be ignored.
#     """)

# Lightning defines batch size as batch size per gpu
assert opt.batch_size % torch.cuda.device_count() == 0
opt.batch_size = opt.batch_size // torch.cuda.device_count()
opt.valid_batch_size = opt.valid_batch_size // torch.cuda.device_count()

# If resume from last checkpoint
# if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, f'{opt.id}_last.ckpt')):
#     resume_from = os.path.join(opt.start_from, f'{opt.id}_last.ckpt')
if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, f'{opt.id}-last.ckpt')):
    resume_from = os.path.join(opt.start_from, f'{opt.id}-last.ckpt')
    if verbose:
        print('resume from', resume_from)
else:
    resume_from = None

from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(
    # project='CLIP-ViL-COCOCaption',
    project='CLIP-Finetune-COCO',
    name=opt.id,
)

if verbose:
    wandb_logger.experiment.config.update(opt)
    from pathlib import Path
    import glob
    import wandb
    # src_dir = Path(__file__).resolve().parent.parent
    glob_str = "*.py"
    base_path = './'
    wandb.save(glob_str=glob_str, base_path=base_path)

    glob_str = "**/*.yaml"
    base_path = './'
    wandb.save(glob_str=glob_str, base_path=base_path)
    
    # code = wandb.Artifact('project-source', type='code')
    # for path in glob.glob('**/*.py', recursive=True):
    #     code.add_file(path, name='source/'+path)
    #     print(path)
    # wandb.run.use_artifact(code)




lit = LitModel(opt)
# warning grad_clip_mode is ignored.
trainer = pl.Trainer(
    callbacks=[
        OnEpochStartCallback(),
        # pl.callbacks.lr_logger.LearningRateLogger()
        pl.callbacks.LearningRateMonitor()
    ],
    default_root_dir=opt.checkpoint_dir,
    resume_from_checkpoint=resume_from,

    distributed_backend='ddp',
    gpus=torch.cuda.device_count(),
    
    # gpus=1,

    check_val_every_n_epoch=1,
    # max_epochs=opt.max_epochs,
    max_epochs=opt.epochs,
    # gradient_clip_val=opt.grad_clip_value,
    gradient_clip_val=opt.clip_grad_norm,
    
    checkpoint_callback=checkpoint_callback,
    log_gpu_memory='min_max',
    # log_save_interval=opt.losses_log_every,
    log_every_n_steps=opt.losses_log_every,
    profiler=True,
    # profiler='simple',
    # row_log_interval=10,  # what is it?
    flush_logs_every_n_steps=10,
    num_sanity_val_steps=0,
    # val_check_interval=0.01,
    # limit_train_batches=500,
    # progress_bar_refresh_rate=0,
    # fast_dev_run=True,
    precision=opt.precision,
    logger=wandb_logger
)

if os.getenv('EVALUATE', '0') == '1':
    trainer.test(lit)
else:
    trainer.fit(lit)
