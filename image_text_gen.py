import torch
import regex as re
from data.prepare import *
from utils.config import *
from model.encode_decode_model import *
from model.text_model import *
from rouge_score import rouge_scorer
from scripts.train_text_gen import *
import torchvision.transforms as transforms

def evaluate_score(gen_text, targets):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gen_text, targets)
    for key in scores:
        print(f'{key}: {scores[key]}')
        
def valid_rouge_score(model, df):
    model.eval()
    transform = transforms.Compose([
        # if using PIL to load image, comment this line
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=1),
    ])
    # just load all valid img & report
    for data in df:
        img = transform(data['image']).to(config.device)
        report = model.generate(img)
        evaluate_score(report, data['text'])

epochs = 1000

def train_image_text(model, optim, train_df, valid_df):
    
    model.to(config.device)
    model.train()
    
    for iter in range(epochs):
    
        for batch in train_df:
            img, targets = batch['image'].to(config.device), batch['text'].to(config.device)
            logits, loss = model(img, targets)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        # break                           # for testing
        
        if iter % 100 == 0:
            print(f'{iter} / {epochs} training loss : {loss}')
            valid_rouge_score(model, valid_df)
        
        if iter % 200 == 0 and iter > 0:
            torch.save(model.state_dict(), f'image_gen_report_for_{iter}_iters.pt')
        
    model.eval()
    valid_rouge_score(model, valid_df)
    torch.save(model.state_dict(), f'text_gen_train_for_{epochs}_iters.pt')


encoder = Encoder(config).to(config.device)
load_checkpoint(encoder, 'your checkpoint')

decoder = Decoder(config).to(config.device)
load_checkpoint(decoder, 'your checkpoint')
model = EncoderDecoderModel(encoder=encoder, decoder=decoder, config=config).to(config.device)
optim = torch.optim.AdamW(model.parameters(), lr=config.lr)

'''
train_df = {}
valid_df = {}

train_image_text(model, optim, train_df, valid_df)
'''