import pandas as pd
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import math

def preprocess_text(text):
    # get start token
    text = '[CLS] ' + text
    # brute force to clear unicode
    text = re.sub('_x000D_', ' ', text)
    # clear the list of number etc. & tokenize
    text = re.sub('[0-9]\)|>|-|[0-9]\.|', '', text)
    gptpat = re.compile(r"""\[[C][L][S]]|\n|[:.,]| [L]4+| *3+[rd]+| *4+[th]+| [LR]\'t|[LR]\'t| T[0-9]+|T[0-9]+| [a-zA-Z]/[a-zA-Z]|[a-zA-Z]/[a-zA-Z]| ?\p{L}+| ?\p{N}+""")
    text = re.findall(gptpat, text)
    tokens = []
    # just make sure token is not empty
    for token in text:
        if len(token) > 0:
            tokens.append(token)
    
    # remove  multiple spaces
    def check_token_head(tokens):
        while tokens[0] == ' ':
            tokens = tokens[1:]
            
        return tokens
    
    tokens = check_token_head(tokens)
        
    return tokens

def create_dict(df):
    vocab_dict = {}
    for row in df:
        for token in row:
            vocab_dict[token] = vocab_dict.get(token, 0) + 1
    sorted_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict

# word tokenize (may change other tokenize method)
def build_vocab(word_dict):
    itos = {}
    stoi = {}
    for i, token in enumerate(word_dict.items()):
        stoi[token[0]] = i
        itos[i] = token[0]
    return itos, stoi

class CustomTokenizer():
    
    def __init__(self, word_dict):
        self.word_dict = word_dict
        
        itos, stoi = build_vocab(word_dict)
        self.encode_ = lambda s : [stoi[c] for c in s]
        self.decode_ = lambda l : ''.join([itos[i] for i in l])

    def encode(self, text):
        return self.encode_(text)
    
    def decode(self, tokens):
        return self.decode_(tokens)
    
class CustomReportDataset(Dataset):
    
    def __init__(self, text_df, split='train', word_dict=None):
        assert split in ['train', 'valid', 'test']
        n_samples = {
            'train' : len(text_df) * 0.9,
            'valid' : len(text_df) * 0.1,
            'test' : len(text_df),
        }[split]
        if type(text_df) is not list:
            text_df = list(text_df)
        self.df = random.sample(text_df, int(n_samples))
        self.tokenizer = CustomTokenizer(word_dict)
        
    def __getitem__(self, index) :
        target = self.df[index]
        target = torch.tensor(self.tokenizer.encode(target))
        return target
        
    # def decode(self, tokens):
    def __len__(self):
        return len(self.df)
        
        
def CustomBlockSeq2Batch(df, block_size, batch_size, threshold=50, device=None, target_idx=None):
    
    ### get rid of the sequence that len < threshold
    # n_df = []
    # for idx, data in enumerate(df):
    #     if len(data) >= threshold: n_df.append(data)
    
    # get random batch
    if target_idx == None: target_idx = random.randint(0, len(df) - 1)
    ix = torch.randint(len(df[target_idx]) - block_size, (batch_size, ))
    ix[0] = 0                                 # test for make sure CLS
    x = torch.stack([df[target_idx][i:i+block_size] for i in ix])
    y = torch.stack([df[target_idx][i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

