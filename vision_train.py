import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
                            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        image_files = set(os.listdir(root_dir))
        self.annotations = self.annotations[self.annotations['Image Index'].isin(image_files)]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        labels = self.annotations.iloc[idx, 1].split('|')
        labels = [1 if label in labels else 0 for label in self.class_names]
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

def train_model(model, criterion, optimizer, dataloaders, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)

    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history

def plot_loss(train_loss_history, val_loss_history):
    epochs = range(len(train_loss_history))
    plt.plot(epochs, train_loss_history, 'r', label='Training loss')
    plt.plot(epochs, val_loss_history, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ChestXrayDataset(csv_file='./ChestXray_data/Data_Entry_2017.csv', root_dir='./ChestXray_data/images', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    val_dataset = ChestXrayDataset(csv_file='./ChestXray_data/Data_Entry_2017.csv', root_dir='./ChestXray_data/images', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    dataloaders = {'train': train_loader, 'val': val_loader}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(pretrained=True)
    num_classes = 14
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model, train_loss_history, val_loss_history = train_model(model, criterion, optimizer, dataloaders, num_epochs=100)
    
    torch.save(model.state_dict(), 'trained_chestxray_model.pth')

    plot_loss(train_loss_history, val_loss_history)
