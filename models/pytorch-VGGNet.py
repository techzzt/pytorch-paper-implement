# 라이브러리 

import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary as summary_
from torch.nn import functional as F
import time

##### DataLoader 정의 #####

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.STL10(root = './data', split = 'train', download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle = True)

testset = torchvision.datasets.STL10(root = './data', split = 'test', download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 32, shuffle = False)

##### Model 구현 #####

class VGG_A(nn.Module):
    def __init__(self, n_class: int = 1000, init_weights: bool = True):
        
        super(VGG_A, self).__init__()
        
        self.convnet = nn.Sequential(
            
            # Input Channel (RGB: 3)
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(inplace = True), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(inplace = True), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True), 
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, n_class),
            nn.Softmax(dim = 1)
            )
        
    def forward(self, x: torch.Tensor):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        

model = VGG_A(n_class = 10)
model = model

summary_(model, (3, 224, 224), batch_size = 10)


##### Model Train #####

classes = ('airplance', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

start_time = time.time()

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        inputs, labels = inputs, labels
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            
            runhnig_loss = 0.0
            
print(time.time() - start_time)
print('Finised Train')


##### Performance #####

class_true = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images
        labels = labels
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        label = (predicted == labels).squeeze()
        
        for i in range(4):
            label = labels[i]
            class_true[label] += label[i].item()
            class_total[label] += 1
            
for i in range(10):
    print("Accuracy of %5s : %2d %%" % (classes[i], 100 * class_true[i] / class_total[i]))
