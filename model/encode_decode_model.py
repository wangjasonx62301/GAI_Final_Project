import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import config
from data.prepare import *

class EnFeedForward(nn.Module):
    # input size is (batch_size, encoder_fan_out=14)
    # target : (batch_size, block_size, n_embd)
    def __init__(self, config, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.encoder_fan_out, config.n_embd * config.block_size),
            nn.ReLU(),
            nn.Linear(config.n_embd * config.block_size, config.n_embd),
        )
    
    def forward(self, x):
        return self.net(x)


class EncoderDecoderModel(nn.Module):
    
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.decoder = decoder
        self.decoder.eval()
        self.ffwd = EnFeedForward(config)
        self.config = config
        
    def forward(self, img, targets=None):
        # can only input batch_size = 1, need to improve
        B, C, H, W = img.shape
        img_features, loss_enc = self.encoder(img=img)
        # print(img_features.shape)
        img_features = self.ffwd(img_features).view(B, 1, self.config.n_embd)
        # print(img_features.shape)
        
        out = self.decoder.generate(idx=torch.tensor([config.stoi['[CLS]']], dtype=torch.long, device=config.device).view(1, -1),
                                    img_features=img_features, max_new_tokens=100)
        if targets == None:
            loss = None
        else:
            B, T = out.shape
            if T < self.config.max_padding:
                out = F.pad(out, (0, self.config.max_padding - len(out)), value=self.config.stoi['[PAD]'])
            loss = F.cross_entropy(out, targets)
            
        return out, loss
