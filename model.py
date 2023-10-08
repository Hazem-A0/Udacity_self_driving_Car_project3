import torch
from torch import nn
import torch.nn.functional as F
## Put your Architecture here

## Here is a sample and this one not supposed to work so delete it and put your own
class SelfDrivingModel(nn.Module):
    def __init__(self):
        super().__init__()
        #input = 320*160*3
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2) #158*78*24
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2) #77*37*36
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2) #37*17*48
        self.pool = nn.MaxPool2d(2, 2) #18*8*48
        self.conv4 = nn.Conv2d(48, 64, 3) #16*6*64
        self.conv5 = nn.Conv2d(64, 64, 3) #14*4*64
        self.fc1 = nn.Linear(14*4*64, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
