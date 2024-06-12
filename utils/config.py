import torch
from dataclasses import dataclass
# from model.text_model import *
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.prepare import create_dict, build_vocab

@dataclass
class config():
    # for all
    vocab_size : int = len(create_dict())
    n_embd     : int = 512
    n_head     : int = 32
    lr         : int = 2e-5
    max_iters  : int = 30000
    n_layer    : int = 40
    eval_iters : int = 10
    eval_ival  : int = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint : int = 5000
    # train, valid data
    train_csv_path = './dataset/merged_reports_train.csv'
    valid_csv_path = './dataset/merged_reports_valid.csv'
    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)
    train_image_dir = './CXIRG_Data/train_data/images'
    valid_image_dir = './CXIRG_Data/valid_data/images'
    # for decoder
    block_size : int = 8
    batch_size : int = 8  
    threshold  : int = 50                                       # allow longer report
    itos, stoi = build_vocab(create_dict())
    # for encoder
    batch_size_enc : int = 8
    max_padding    : int = 128
    max_length     : int = 128
    encoder_fan_out: int = 14
    image_size     : int = 224
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=1),
    ])
    scaler = StandardScaler()
    scaler.fit(train_df[['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12', 'class_13', 'class_14']])
