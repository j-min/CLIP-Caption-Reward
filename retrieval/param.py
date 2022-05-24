import argparse
import random

import numpy as np
import torch

import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        # optimizer = torch.optim.AdamW
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Data Splits
    parser.add_argument("--train", default='karpathy_train')
    parser.add_argument("--valid", default='karpathy_val')
    parser.add_argument("--test", default='karpathy_test')
    # parser.add_argument('--test_only', action='store_true')

    # Quick experiments
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)

    # Checkpoint
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--from_scratch', action='store_true')

    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=-1)
    # parser.add_argument('--rank', type=int, default=-1)

    # Model Config
    # parser.add_argument('--encoder_backbone', type=str, default='openai/clip-vit-base-patch32')
    # parser.add_argument('--decoder_backbone', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer', type=str, default='openai/clip-vit-base-patch32')

    # parser.add_argument('--position_embedding_type', type=str, default='absolute')

    # parser.add_argument('--encoder_transform', action='store_true')

    parser.add_argument('--max_text_length', type=int, default=40)

    # parser.add_argument('--image_size', type=int, default=224)
    # parser.add_argument('--patch_size', type=int, default=32)

    # parser.add_argument('--decoder_num_layers', type=int, default=12)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=None)

    parser.add_argument('--optim', default='adamw')

    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)

    parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--dropout', type=float, default=0.1)


    # Inference
    # parser.add_argument('--num_beams', type=int, default=1)
    # parser.add_argument('--gen_max_length', type=int, default=20)

    parser.add_argument('--start_from', type=str, default=None)

    # Data
    # parser.add_argument('--do_lower_case', type=str2bool, default=None)

    # parser.add_argument('--prefix', type=str, default=None)


    # COCO Caption
    # parser.add_argument('--no_prefix', action='store_true')

    parser.add_argument('--no_cls', action='store_true')

    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--id', type=str, default=None)

    # Etc.
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    loaded_kwargs = {}
    if args.cfg is not None:
        cfg_path = f'configs/{args.cfg}.yaml'
        with open(cfg_path, 'r') as f:
            loaded_kwargs = yaml.safe_load(f)

    # Namespace => Dictionary
    parsed_kwargs = vars(args)
    parsed_kwargs.update(optional_kwargs)

    kwargs = {}
    kwargs.update(parsed_kwargs)
    kwargs.update(loaded_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    # def update(self, **kwargs):
    #     for k, v in kwargs.items():
    #         setattr(self, k, v)

    # def save(self, path):
    #     with open(path, 'w') as f:
    #         yaml.dump(self.__dict__, f, default_flow_style=False)

    # @classmethod
    # def load(cls, path):
    #     with open(path, 'r') as f:
    #         kwargs = yaml.load(f)

    #     return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
