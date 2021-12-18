import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.optim import Adam, SGD


#Task 1
class Convolutional_Neural_Network(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10):
        super(Convolutional_Neural_Network, self).__init__()

        #now the hidden layers
        #hidden layer 1
        self.convolutional_layer1 = nn.Conv2d(kernel_size = [7, 7], stride = 2, padding = 3, in_channels = 3, out_channels = 64)

        #applying batch normalisation on our first hidden layer
        self.convolutional_layer1_batch = nn.BatchNorm2d(64)

        #applying the relu function on our first hidden layer
        self.convolutional_layer1_relu = nn.ReLU(inplace = True)
        
        #hidden layer 2
        self.convolutional_layer2 = nn.MaxPool2d(kernel_size = [3, 3], stride = 2, padding = 0)

        #hidden layer 3
        self.convolutional_layer3 = nn.Conv2d(kernel_size = [3, 3], stride = 1, padding = 1, in_channels = 3, out_channels = 64)

        #batch normalization layer
        self.convolutional_layer3_batch = nn.BatchNorm2d(64)
        
        #relu layer
        self.convolutional_layer3_relu = nn.ReLU(inplace = True)
        
        #hidden layer 4
        self.convolutional_layer4 = nn.MaxPool2d(kernel_size = [3, 3], stride = 2, padding = 0)

        #hidden layer 5
        self.fully_connected_layer = nn.Linear(in_features = 5, out_features = 5)

        #softmax layer
        self.softmax = nn.Softmax(dim=None)

#define and print the model
cnn = Convolutional_Neural_Network()
print(cnn)

#Task 2
#define the optimizer
#CHANGE LR and speak abour it (lr = learning rate)
#try 0.01, 0.001, 0.0001, 0.00001 then discuss
optimizer = Adam(cnn.parameters(), lr = 0.01)
#define the loss function
loss = CrossEntropyLoss()

#check GPU is able to work
if torch.cuda.is_available():
    cnn = cnn.cuda()
    loss = loss.cuda()

