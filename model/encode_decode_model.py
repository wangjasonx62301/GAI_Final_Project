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
        self.tokenizer = CustomTokenizer(create_dict())
        
    def forward(self, img, targets=None):
        # can only input batch_size = 1, need to improve
        B, C, H, W = img.shape
        img_features, loss_enc = self.encoder(img=img)
        # print(img_features.shape)
        img_features = self.ffwd(img_features).view(B, 1, self.config.n_embd)
        # print(img_features.shape)
        
        out = self.decoder.generate(idx=torch.tensor([config.stoi['[CLS]']], dtype=torch.long, device=config.device).view(1, -1),
                                    img_features=img_features, max_new_tokens=64)
        if targets == None:
            loss = None
        else:
            if type(targets) == str:
                targets = torch.tensor(self.tokenizer.encode(preprocess_text(targets))).to(self.config.device).to(torch.int64)
                if len(targets) < self.config.max_padding:
                    targets = F.pad(targets, (0, self.config.max_padding - len(targets)), value=self.config.stoi['[PAD]'])
                elif len(targets) > self.config.max_padding:
                    targets = targets[:self.config.max_padding]
                    
            B, T = out.shape
            if T < self.config.max_padding:
                out = F.pad(out, (0, self.config.max_padding - len(out)), value=self.config.stoi['[PAD]']).to(torch.int64)
            loss = F.cross_entropy(out, targets)
            
        return out, loss
    
    def generate(self, img):
        out, loss = self(img, targets=None)
        decode = lambda l : ''.join([config.itos[i] for i in l])
        report = (decode(out[0].tolist()))
        return re.sub('\[CLS\]', '', report)