# our CNN that we need to create and train
# https://www.geeksforgeeks.org/building-a-convolutional-neural-network-using-pytorch/ for an example
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class ColorizationNet(nn):
    def __init__(self, num_classes=10):

    def forward(self, x):
        # example of forward pass from previous assignment
        #out = self.layer1(x)
        #out = self.layer2(out)
        #out = out.reshape(out.size(0), -1)
        #out = self.fc(out)
        #return out


# step 4 on geeks for geeks is very important for backward propagation