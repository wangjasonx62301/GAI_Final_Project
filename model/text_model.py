import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from utils.config import config


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
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        
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
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T)
        for _ in range(max_new_tokens):
            # get predictions
            idx_cond = idx[:, -self.block_size:] # prevent longer block_size, because we just have pos. embd
            logits, loss = self(idx_cond) # now (B, T, C)
            logits = logits[:, -1, :] # now get the last step and shape (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    

class Encoder(nn.Module):
    
    def __init__(self, config, sinusoidal_embedding=False):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(config.vocab_size, config.n_embd)
        if sinusoidal_embedding == True:
            self.position_embedding_table = SinusoidalEmbedding(config.max_padding, config.n_embd)
        else : self.position_embedding_table = torch.nn.Embedding(config.max_padding, config.n_embd)
        # make sure encoder term
        self.blocks = nn.Sequential(*[Block(config, encoder_term=True) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.encoder_fan_out)
        self.device = config.device
        self.block_size = config.max_padding
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss