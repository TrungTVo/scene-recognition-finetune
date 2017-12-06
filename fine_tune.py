
####################
# Author : Trung Vo
####################
import torch
from torchvision import datasets, models, transforms
from torchvision import models

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import cv2
import numpy as np
import glob
import random 
import math
import cv2
import time
import copy
import os
#import matplotlib.pyplot as plt


# In[4]:

# ==========================================
#    Load Training Data and Testing Data
# ==========================================

#class_names = [name[11:] for name in glob.glob('data/train/*')]
#class_names = dict(zip(range(len(class_names)), class_names))
#print (class_names)


# load dataset
def load_dataset(img_size = (64,64)):
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([     
            transforms.Scale(img_size),   
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([   
            transforms.Scale(img_size),   
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=0) for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    return dataset_sizes, dataloaders


# laod dataset
img_size = (64,64)
dataset_sizes, dataloaders = load_dataset(img_size = img_size)


# define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input size = 64x64x1
        self.conv1 = nn.Conv2d(3, 6, 5, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=1)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(14 * 14 * 16, 120)
        self.fc2 = nn.Linear(120, 84) 
        #self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(84, 15)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 14 * 14 * 16)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        #x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x)


net = Net()

# use gpu or not
use_gpu = torch.cuda.is_available()

gpu_id = 1

# use gpu or not
if use_gpu:
    net = net.cuda(gpu_id)



# train model 
def train_model(model, criterion, optimizer, num_epochs):
    since = time.time()   # start time

    best_model = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))            

        running_loss = 0.0
        running_corrects = 0
        n_batches = len(dataloaders['train'])

        # Iterate over data.
        for data in dataloaders['train']:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda(gpu_id))
                labels = Variable(labels.cuda(gpu_id))
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # backward + optimize 
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

        # final loss and accuracy
        epoch_loss = running_loss / n_batches
        epoch_acc = float(running_corrects) / float(dataset_sizes['train'])

        if epoch % 5 == 0:
            print('train loss: {:.4f} \t train accuracy: {:.4f} %'.format(epoch_loss, epoch_acc*100))
            print('-' * 50)

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best train accuracy: {:4f} %'.format(best_acc*100))

    # load best model weights
    model.load_state_dict(best_model)
    return model


# evaluate on test set
def evaluate(model):
    running_loss = 0.0
    running_corrects = 0
    n_batches = len(dataloaders['test'])

    # Iterate over data.
    for data in dataloaders['test']:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda(gpu_id))
            labels = Variable(labels.cuda(gpu_id))
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        
        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    # final loss and accuracy
    epoch_loss = running_loss / n_batches
    epoch_acc = float(running_corrects) / float(dataset_sizes['test'])    
    print ('Evaluate .....')
    print('Best test accuracy: {:4f} %'.format(epoch_acc*100))

# Train from scratch
'''
LEARNING_RATE = 0.001

if use_gpu:
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# train model
#model = train_model(net, criterion, optimizer, num_epochs=41)
#evaluate(model)
'''

# Fine-tune AlexNet
# laod dataset
img_size = (224,224)
dataset_sizes, dataloaders = load_dataset(img_size = img_size)

# Pretrain AlexNet
model_ft = models.alexnet(pretrained=True)

list_of_layers = list(model_ft.classifier.children())
last_layer_in_features = list_of_layers[len(list_of_layers)-1].in_features
#print ('\nnumber of inputs last layer:', last_layer_in_features)
del list_of_layers[len(list_of_layers)-1]
list_of_layers.append(nn.Linear(last_layer_in_features, 15 ))

# created new model
model_ft.classifier = nn.Sequential(*list_of_layers)

if use_gpu:
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
else:
    criterion = nn.CrossEntropyLoss()

# Observe that only parameters in last layer are being optimized
LEARNING_RATE = 0.001
#optimizer_ft = torch.optim.SGD(model_ft.classifier[len(model_ft.classifier)-1].parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

if use_gpu:
    model_ft = model_ft.cuda(gpu_id)

# train model
model = train_model(model_ft, criterion, optimizer_ft, num_epochs=91)
evaluate(model)

