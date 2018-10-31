import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import os

class TransferLearning:
    
    def __init__(self, data_dir, batch_size=4):
        
         # transformations
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.tfm_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    
        self.tfm_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        
        # data loaders
        
        self.ds_train = ImageFolder(data_dir+'/train', self.tfm_train)
        self.ds_val = ImageFolder(data_dir+'/val', self.tfm_val)
        
        self.dl_train = DataLoader(self.ds_train, batch_size=batch_size, shuffle = True, num_workers=4)
        self.dl_val = DataLoader(self.ds_val, batch_size=batch_size, shuffle = True, num_workers=4)
    
        # device
        self.device = torch.device("cpu")
            
        # build model
        
        model = torchvision.models.resnet18(pretrained=True)

        for p in model.parameters():
            p.requires_grad = False
            
        nf = model.fc.in_features
        
        model.fc = nn.Linear(nf, 2)
        model = model.to(self.device)
        
        self.model = model
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)


    def train_model(self, num_epochs=25):

        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        device = self.device
        dl_train = self.dl_train
        ds_train = self.ds_train
        dl_val = self.dl_val
        ds_val = self.ds_val

        # for each epoch
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            #train

            model.train()
            running_loss=0.0
            running_corrects = 0

            #for each minibatch
            for inputs, labels in dl_train:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)

            epoch_loss = running_loss / len(ds_train)
            epoch_acc = running_corrects.double() / len(ds_train)

            print('TRAIN LOSS: {:.4f} ACC: {:.4f}'.format(epoch_loss, epoch_acc))

            #eval

            model.eval()
            running_loss=0.0
            running_corrects = 0

            for inputs, labels in dl_val:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds==labels.data)

            epoch_loss = running_loss / len(ds_train)
            epoch_acc = running_corrects.double() / len(ds_train)

            print('VALIDATION LOSS: {:.4f} ACC: {:.4f}'.format(epoch_loss, epoch_acc))