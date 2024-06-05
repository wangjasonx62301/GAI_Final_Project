import torch
from data.prepare import *
from utils.config import *

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    word_dict = create_dict()
    for split in ['train', 'valid']:
        losses = torch.zeros(config.eval_iters)
        df = CustomReportDataset(split, word_dict=word_dict)
        for k in range(config.eval_iters):
            if split == 'valid': 
                X, Y = CustomBlockSeq2Batch(df, config, valid=True)
            else : X, Y = CustomBlockSeq2Batch(df, config, valid=False)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    return out

def train_gen(model, optim):
    
    df = CustomReportDataset(split='train')
    
    model.train()
    
    for iter in range(config.max_iters):
    
        if iter % config.eval_ival == 0:
            losses = estimate_loss(model)
            print(f"step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")
        
        xb, yb = CustomBlockSeq2Batch(df, config, target_idx=iter%len(df))
        logits, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        # break                           # for testing
    
    model.eval()
    
def generate_report(model, context=None):
    
    encode = lambda s : [config.stoi[c] for c in s]
    decode = lambda l : ''.join([config.itos[i] for i in l])
    if context == None: context = '[CLS]'
    context = preprocess_text(context)
    context_tokens = encode(context)
    context = torch.tensor([context_tokens], dtype=torch.long, device=config.device)
    print(decode(model.generate(context, max_new_tokens=50)[0].tolist()))
