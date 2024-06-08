import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.prepare import CustomTokenizer, create_dict, preprocess_text
from utils.config import config
from model.text_model import Encoder

class CombinedDataset(Dataset):
    def __init__(self, data, tokenizer, scaler, max_length=128):
        self.image_features = data[['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12', 'class_13', 'class_14']].values
        self.texts = data['text'].values
        self.labels = data[['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12', 'class_13', 'class_14']].values
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.max_length = max_length
        self.scaled_image_features = self.scaler.transform(self.image_features)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        image_features = torch.tensor(self.scaled_image_features[idx], dtype=torch.float)
        text = self.texts[idx]
        encoded_text = self.tokenizer.encode(preprocess_text(text))
        if len(encoded_text) < self.max_length:
            encoded_text += [0] * (self.max_length - len(encoded_text))  # padding
        else:
            encoded_text = encoded_text[:self.max_length]
        text_features = torch.tensor(encoded_text, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (image_features, text_features), label

train_file_path = './dataset/merged_reports_train.csv'
valid_file_path = './dataset/merged_reports_valid.csv'
train_data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)

scaler = StandardScaler()
scaler.fit(train_data[['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12', 'class_13', 'class_14']])

vocab_dict = create_dict()
tokenizer = CustomTokenizer(vocab_dict)

train_dataset = CombinedDataset(train_data, tokenizer, scaler)
valid_dataset = CombinedDataset(valid_data, tokenizer, scaler)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

model = Encoder(config=config)
model.to(config.device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
criterion = torch.nn.CrossEntropyLoss()

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(config.max_iters):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        (image_features, text_features), labels = batch
        print(labels.shape)
        image_features, text_features, labels = image_features.to(config.device), text_features.to(config.device), labels.to(config.device)

        optimizer.zero_grad()
        # inputs = torch.cat((image_features, text_features), dim=1)
        inputs = text_features
        # print(inputs.shape)
        outputs, loss = model(inputs, targets=labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(train_loader))
    
    if epoch % config.eval_ival == 0:
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                (image_features, text_features), labels = batch
                image_features, text_features, labels = image_features.to(config.device), text_features.to(config.device), labels.to(config.device)
                inputs = torch.cat((image_features, text_features), dim=1)
                outputs, loss = model(inputs, targets=labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
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
