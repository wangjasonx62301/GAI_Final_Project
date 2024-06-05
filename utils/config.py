import torch
from dataclasses import dataclass
from model.text_model import *
from data.prepare import *


@dataclass
class config():
    vocab_size : int = len(create_dict())
    block_size : int = 8
    batch_size : int = 8
    n_embd     : int = 512
    n_head     : int = 32
    lr         : int = 2e-5
    max_iters  : int = 10000
    n_layer    : int = 40
    eval_iters : int = 10
    eval_ival  : int = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold  : int = 50                                       # allow longer report
    itos, stoi = build_vocab(create_dict())
