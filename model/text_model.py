import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from utils.config import config
import regex as re

class SinusoidalEmbedding(nn.Module):
    
    def __init__(self, block_size, n_embd):
        super().__init__()
        self.emb_wei = torch.zeros(block_size, n_embd)
        wei = torch.tensor([1 / 10000 ** (2 * j / n_embd) for j in range(n_embd)]).view(1, n_embd)
        t = torch.arange(block_size).view(block_size, 1)
        # even idx embedding
        self.emb_wei[:, ::2] = torch.sin(t * wei[:, ::2])
        self.emb_wei[:, 1::2] = torch.cos(t * wei[:, ::2])
        
        self.embedding = nn.Embedding(block_size, n_embd)
        self.embedding.weight.data = self.emb_wei
        
    def forward(self, x):
        out = self.embedding(x)
        return out

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config, encoder_term=False):
        super().__init__()
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.encoder_term = encoder_term
        
    def forward(self, x):
        
        # batch_size, Seq_len, embedding dim
        B, T, C = x.shape
        # print(x.shape)
        # after c_attn(x), the shape is B, T, n_embd * 3
        a = self.c_attn(x)
        q, k, v = a.split(self.n_embd, dim=2)
        # start view() & transpose()
        # shape after transpose (Batch_size, n_head, Seq_len, n_embd // n_head) 
        # or (B, n_head, T, C // n_head)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(2, 1)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(2, 1)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(2, 1)
        # the formula : softmax(QK^T / sqrt(embd_dim(k)))V
        # shape after q @ k : (B, n_head, T, T) 
        attn = q @ k.transpose(-2, -1) * (1 / math.sqrt(self.n_embd * 3 // self.n_head))
        # encoder
        if self.encoder_term == False:
            attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        # shape after attn @ v : (B, n_head, T, C // n_head)
        y = attn @ v
        y = y.transpose(2, 1).contiguous().view(B, T, C)
        self.out = self.c_proj(y)
        return self.out   
    
class FeedForward(nn.Module):
    
    def __init__(self, config, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    
    def __init__(self, config, encoder_term=False):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, encoder_term=encoder_term)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        
    def forward(self, x):
        # x shape (B, T, C)
        x = x + self.sa(self.ln1(x))        # (B, T, C)
        x = x + self.ffwd(self.ln2(x))      # (B, T, C)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, config, sinusoidal_embedding=False):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(config.vocab_size, config.n_embd)
        if sinusoidal_embedding == True:
            self.position_embedding_table = SinusoidalEmbedding(config.block_size, config.n_embd)
        else : self.position_embedding_table = torch.nn.Embedding(config.block_size, config.n_embd)
        
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.device = config.device
        self.block_size = config.block_size
        
    def forward(self, idx, targets=None, img_features=None):
        
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) (8, 8, 512)
        if img_features != None:
            x += img_features
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets == None:
            loss = None
        else:
            
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, img_features=None):
        # idx is (B, T)
        for _ in range(max_new_tokens):
            # get predictions
            idx_cond = idx[:, -self.block_size:] # prevent longer block_size, because we just have pos. embd
            logits, loss = self(idx=idx_cond, img_features=img_features) # now (B, T, C)
            logits = logits[:, -1, :] # now get the last step and shape (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def generate_report(self, context=None, preprocess_text=None, img_features=None):
    
        encode = lambda s : [config.stoi[c] for c in s]
        decode = lambda l : ''.join([config.itos[i] for i in l])
        if context == None: context = ''
        context = preprocess_text(context)
        context_tokens = encode(context)
        context = torch.tensor([context_tokens], dtype=torch.long, device=config.device)
        report = (decode(self.generate(context, max_new_tokens=64, img_features=img_features)[0].tolist()))
        return re.sub('\[CLS\]', '', report)    

class FeatureExtracter(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 14 * 14, 128 * self.config.n_embd), 
            nn.ReLU()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_layers(x)
        x = x.view(B, 1, -1)  
        x = self.fc_layers(x)
        x = x.view(B, 128, self.config.n_embd)  
        return x

class Encoder(nn.Module):
    
    def __init__(self, config, sinusoidal_embedding=False):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(config.vocab_size, config.n_embd)
        if sinusoidal_embedding == True:
            self.position_embedding_table = SinusoidalEmbedding(config.max_padding, config.n_embd)
        else : self.position_embedding_table = torch.nn.Embedding(config.max_padding, config.n_embd)
        self.fx = FeatureExtracter(config)
        # make sure encoder term
        self.blocks = nn.Sequential(*[Block(config, encoder_term=True) for _ in range(config.n_layer)])
        self.lm_head = nn.Sequential(
            nn.Linear(config.n_embd, config.encoder_fan_out),
            nn.ReLU(),
        )
        self.l_out = nn.Linear(config.max_padding * config.encoder_fan_out, config.encoder_fan_out)
        self.device = config.device
        self.block_size = config.max_padding
        
    def forward(self, idx=None, img=None, targets=None):
        # make sure whole img is input, and img requires transform
        
        imf = self.fx(img)
        if idx != None:
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
            x = 0.1 * (tok_emb + pos_emb) + 0.9 * imf # (B, T, C)
        else: 
            B, T, C = imf.shape
            x = imf
        x = self.blocks(x)
        x = self.lm_head(x).view(B, -1) # (B, T * fan_out(14))
        logits = self.l_out(x)          # (B, fan_out(14))
        if targets == None:
            loss = None
        else:
            loss = F.mse_loss(logits, targets)
        
        return logits, loss