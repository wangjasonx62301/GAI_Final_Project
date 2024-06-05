import torch
import torch.nn as nn
import torch.nn.functional as F
from model.text_model import *
from data.prepare import *
from utils.config import *
from scripts.train_text_gen import *

if __name__ == '__main__':

    print('----------load model----------')
    model = Decoder(config)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
    print('--------finish loading--------')
    print('--------start training--------')
    train_gen(model, optim)
    generate_report(model)

