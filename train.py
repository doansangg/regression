import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR100

import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from dataloader import get_loader_train
from configs import *

train_data_loader,train_data_size = get_loader_train("./bsx_data",bs)
test_data_loader,test_data_size = get_loader_train("./data_real",bs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet50 Model
resnet50 = models.resnet50(pretrained=True)
resnet50 = resnet50.to(device)
summary(resnet50, (3, 360, 360))
# Freeze layers 1-6 in total 10 layers of Resnet50
ct = 0
for child in resnet50.children():
  ct += 1
  if ct < 7:
      for param in child.parameters():
          param.requires_grad = False

# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = resnet50.fc.in_features

resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 8), # Since 100 possible outputs
    # nn.LogSoftmax(dim=1) # For using NLLLoss()
)

# Convert model to be used on GPU
resnet50 = resnet50.to(device)

# Define Optimizer and Loss Function
loss_func = nn.MSELoss()


def train_and_validate(model, loss_criterion, optimizer, scheduler,epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    start = time.time()
    history = []
    best_loss_train = 100000.0
    best_loss_val = 100000.0
    best_epoch = None

    for epoch in range(epochs):
        epoch_start = time.time()
        #print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):
            #print(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            scheduler.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

        
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(test_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # if not j%100:
                #   print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}".format(j, loss.item()))

        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/test_data_size 

        #history.append([avg_train_loss, avg_valid_loss])
                
        epoch_end = time.time()
        #torch.save(model, 'model.pt')
        print("Epoch : {:3d}, Training-Loss: {:.4f}, Validation-Loss: {:.4f}, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_valid_loss, epoch_end-epoch_start)) 
        if avg_valid_loss <= best_loss_val:
            best_loss_val = avg_valid_loss
            best_epoch = epoch
            # Save if the model has best accuracy till now
            torch.save(model, 'last.pt')
            print("---------------------Save Loss---------------------Best valid loss: ",best_loss_val) 
            if avg_train_loss <= best_loss_train:
                best_loss_train = avg_train_loss
                torch.save(model, 'best.pt')
                print("------------Save Best Train Val Loss-----------Best train loss: ",best_loss_train) 
    return model, history, best_epoch

optimizer = optim.Adam(resnet50.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,350], gamma=0.1)
trained_model, history, best_epoch = train_and_validate(resnet50, loss_func, optimizer, scheduler, num_epochs)
#torch.save(history, 'history.pt')
