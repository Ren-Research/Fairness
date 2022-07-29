#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        
#        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
#        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
                                    
    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        
#        x = self.layer_hidden1(x)
#        x = self.layer_hidden(x)
        return x
    
    


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args, rate=1, kernel=5):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, int(6*rate), kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(int(6*rate), int(16*rate), 5)
        self.fc1 = nn.Linear(int(16*rate) * 5 * 5, int(120*rate))
        self.fc2 = nn.Linear(int(120*rate), int(84*rate))
        self.fc3 = nn.Linear(int(84*rate), args.num_classes)
        
        self.rate = rate

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, int(16*self.rate) * 5 * 5)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class cnn(nn.Module):
    def __init__(self, args, rate=1, kernel=5):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, int(32*rate), kernel_size=kernel)
        size1 = (32 - kernel + 2*0) / 1 + 1 # input image size: 32, after conv2d
        self.pool = nn.MaxPool2d(2, 2)
        size2 = (size1 - 2 + 2*0) // 2 + 1   # after pooling
        self.conv2 = nn.Conv2d(int(32*rate), int(32*rate), kernel_size=kernel)
        size3 = (size2 - kernel + 2*0) / 1 + 1  # after conv2d
        size4 = (size3 - 2 + 2*0) // 2 + 1   # after pooling
        size5 = int(32*rate) * size4 * size4    # input for fc1
        #print(size1, size2, size3, size4, size5)
        self.fc1 = nn.Linear(int(size5), int(256*rate))
        self.fc2 = nn.Linear(int(256*rate), 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    