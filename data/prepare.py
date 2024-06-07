import pandas as pd
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

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

def load_tokenized_text(split=None, path=None):
    
    if path == None and split==None: path = 'CXIRG_Data\\train_data\\reports.xlsx'
    elif path == None:
        assert split in [None, 'train', 'valid']
        path = {
            'train' : 'CXIRG_Data\\train_data\\reports.xlsx',
            'valid' : 'CXIRG_Data\\valid_data\\reports.xlsx'
        }[split]
    report = pd.read_excel(path, engine='openpyxl')
    report_texts = report['text'].apply(preprocess_text)
    return report_texts


def create_dict():
    vocab_dict = {}
    for split in ['train', 'valid']:
        df = load_tokenized_text(split)
        for row in df:
            for token in row:
                vocab_dict[token] = vocab_dict.get(token, 0) + 1
    vocab_dict['[PAD]'] = 0
    sorted_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
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
    
    def __init__(self, split='train', word_dict=None, encoder_term=False, config=None):
        assert split in ['train', 'valid', 'test']
        path = {
            'train' : 'CXIRG_Data\\train_data\\reports.xlsx',
            'valid' : 'CXIRG_Data\\valid_data\\reports.xlsx',
            'test'  : 'unknown',
        }[split]
        text_df = load_tokenized_text(path=path)
        if type(text_df) is not list:
            text_df = list(text_df)
        if word_dict is None:
            word_dict = create_dict()
        self.df = text_df
        self.tokenizer = CustomTokenizer(word_dict)
        self.encoder_term = encoder_term
        self.config = config
        
    def __getitem__(self, index) :
        target = self.df[index]
        target = torch.tensor(self.tokenizer.encode(target))
        if self.encoder_term == True:
            if len(target) < self.config.max_padding:
                target = F.pad(target, (0, self.config.max_padding - len(target)), value=self.config.stoi['[PAD]'])
            else: target = target[:self.config.max_length]
        return target
        
    # def decode(self, tokens):
    def __len__(self):
        return len(self.df)
        
        
def CustomBlockSeq2Batch(df, config, target_idx=None, valid=False):
    
    ### get rid of the sequence that len < threshold
    if valid == False:
        n_df = []
        for idx, data in enumerate(df):
            if len(data) >= config.threshold: n_df.append(data)
        df = n_df
    
    # get random batch
    if target_idx == None: target_idx = random.randint(0, len(df) - 1)
    ix = torch.randint(len(df[target_idx]) - config.block_size, (config.batch_size, ))
    # ix[0] = 0                                 # test for make sure CLS
    x = torch.stack([df[target_idx][i:i+config.block_size] for i in ix])
    y = torch.stack([df[target_idx][i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

