import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from data.prepare import *
from utils.config import *
from model.text_model import Encoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def train_encoder(model):
    # prepare dataset
    vocab_dict = create_dict()
    tokenizer = CustomTokenizer(vocab_dict)

    train_dataset = CombinedDataset(config.train_df, tokenizer, img_dir='./CXIRG_Data/train_data/images', config=config)
    valid_dataset = CombinedDataset(config.valid_df, tokenizer, img_dir='./CXIRG_Data/valid_data/images', config=config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    model = Encoder(config=config)
    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(config.max_iters): # 
        train_losses = []
        val_losses = []
        val_accuracies = []
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            (images, text_features), labels = batch
            images, text_features, labels = images.to(config.device), text_features.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            outputs, loss = model(idx=text_features, img=images, targets=labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # break                       # for testing
        
        train_losses.append(running_loss / len(train_loader))
        # break                                                   # for testing
        
        if epoch % config.eval_ival == 0:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    (images, text_features), labels = batch
                    images, text_features, labels = images.to(config.device), text_features.to(config.device), labels.to(config.device)
                    outputs, loss = model(idx=text_features, img=images, targets=labels)
                    val_loss += loss.item()
                    # _, predicted = torch.max(outputs.data, 1)
                    predicted = (outputs > 0.5).float()
                    total += labels.numel()
                    correct += (predicted == labels).sum().item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(correct / total)
            print(f'Epoch {epoch}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {correct / total}')
        
        if epoch % config.checkpoint == 0:
            torch.save(model.state_dict(), f'model_checkpoint_{epoch}.pt')

    print('Finished')

    torch.save(model.state_dict(), 'final_model.pt')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('training_curves.png')
    plt.show()
