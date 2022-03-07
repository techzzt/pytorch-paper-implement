# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:07:11 2022

@author: keun
"""

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from resnext import ResNeXt50, ResNeXt101, ResNeXt152
import os
import torchvision.models as models

# Learning Rate Scheduler 
def lr_scheduler(optimizer, epoch):
    lr = learning_rate
    if epoch >= 150:
        lr /= 10
    if epoch >= 225:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    
# Xavier 
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform(m.weight)
        
### DataLoader ###
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])

train_dataset = torchvision.datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform_train)
test_dataset = torchvision.datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform_test)

train_loader = torch.utils.DataLoader(train_dataset, batch_size = 256, shuffle = True, num_workers = 8)
test_loader = torch.utils.DataLoader(test_dataset, batch_size = 256, shuffle = False, num_workers = 8)

# load model 
device = 'cuda'
model = ResNeXt50()
model.apply(init_weights)
model = model.to(device)

# Traininig
learning_rate = 0.005
num_epooch = 250
model_name = "model.pth"

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 0.0001)

train_loss = 0
valid_loss = 0
correct = 0
total_cnt = 0
best_acc = 0

# Stepwise learning
for epoch in range(num_epoch):
    print(f"========= { epoch + 1} epoch of { num_epoch } =========")
    model.train()
    lr_scheduler(optimizer, epoch)
    train_loss = 0
    valid_loss = 0
    correct = 0
    total_cnt = 0
    
    # train phase
    for step, batch in enumerate(train_loader):
        batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        
        logits = model(batch[0])
        loss = loss_fn(logits, batch[1])
        loss.backward()
        
        optimizer.step()
        train_loss += loss.item()
        _, predict = logits.max(1)
        
        total_cnt += batch[1].size(0)
        correct += predict.eq(batch[1]).sum().item()
        
        if step % 100 == 0 and step != 0:
            print(f"\n====== { step } Step of { len(train_loader) } ======")
            print(f"Train Acc : { correct / total_cnt }")
            print(f"Train Loss : { loss.item() / batch[1].size(0) }")
            
    correct = 0
    total_cnt = 0
    
    # test phase
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(test_loader):
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model(batch[0])
            valid_loss += loss_fn(logits, batch[1])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
        valid_acc = correct / total_cnt
        print(f"\nValid Acc : { valid_acc }")
        print(f"Valid Loss : { valid_loss / total_cnt }")
        
        if (valid_acc > best_acc):
            best_acc = valid_acc
            torch.save(model, model_name)
            print("Model Saved")