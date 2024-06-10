import torch
import regex as re
from data.prepare import *
from utils.config import *
from model.encode_decode_model import *
from model.text_model import *
from rouge_score import rouge_scorer
from scripts.train_text_gen import *
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

def evaluate_score(gen_text, targets):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gen_text, targets)
    for key in scores:
        print(f'{key}: {scores[key]}')
        
def valid_rouge_score(model, df, image_directory):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=1),
    ])
    for index, row in df.iterrows():
        image_path = f"{image_directory}/{row['name']}_1_1.png"
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(config.device)
        report = model.generate(img)
        evaluate_score(report, row['text'])

epochs = 1000

def train_image_text(model, optim, train_df, valid_df, train_image_dir, valid_image_dir):
    model.to(config.device)
    model.train()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=1),
    ])
    
    for iter in range(epochs):
        for index, row in train_df.iterrows():
            image_path = f"{train_image_dir}/{row['name']}_1_1.png"
            img = Image.open(image_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(config.device)
            targets = row['text']
            logits, loss = model(img, targets)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        
        if iter % 100 == 0:
            print(f'{iter} / {epochs} training loss : {loss}')
            valid_rouge_score(model, valid_df, valid_image_dir)
        
        if iter % 200 == 0 and iter > 0:
            torch.save(model.state_dict(), f'image_gen_report_for_{iter}_iters.pt')
        
    model.eval()
    valid_rouge_score(model, valid_df, valid_image_dir)
    torch.save(model.state_dict(), f'text_gen_train_for_{epochs}_iters.pt')

train_csv_path = './dataset/merged_reports_train.csv'
valid_csv_path = './dataset/merged_reports_valid.csv'

train_df = pd.read_csv(train_csv_path)
valid_df = pd.read_csv(valid_csv_path)

train_image_dir = './CXIRG_Data/train_data/images'
valid_image_dir = './CXIRG_Data/valid_data/images'

encoder = Encoder(config).to(config.device)
load_checkpoint(encoder, './encoder.pt')

decoder = Decoder(config).to(config.device)
load_checkpoint(decoder, './decoder.pt')
model = EncoderDecoderModel(encoder=encoder, decoder=decoder, config=config).to(config.device)
optim = torch.optim.AdamW(model.parameters(), lr=config.lr)

train_image_text(model, optim, train_df, valid_df, train_image_dir, valid_image_dir)
