import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_head, n_embd):
        super().__init__()
        
        self.n_embd = n_embd
        self.n_head = n_head
        
        self.c_attn = nn.Linear(n_embd, n_embd * 3)
        self.c_proj = nn.Linear(n_embd, n_embd)
        
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
        attn = F.softmax(attn, dim=-1)
        # shape after attn @ v : (B, n_head, T, C // n_head)
        y = attn @ v
        y = y.transpose(2, 1).contiguous().view(B, T, C)
        self.out = self.c_proj(y)
        return self.out   
    
class FeedForward(nn.Module):
    
    def __init__(self, n_embd, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        
    def forward(self, x):
        # x shape (B, T, C)
        x = x + self.sa(self.ln1(x))        # (B, T, C)
        x = x + self.ffwd(self.ln2(x))      # (B, T, C)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, vocab_size, block_size, n_embd, n_head, device, n_layer=8):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        self.block_size = block_size
        
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
    
        