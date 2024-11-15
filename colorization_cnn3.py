# our CNN that we need to create and train
# Example of CNN : https://www.geeksforgeeks.org/building-a-convolutional-neural-network-using-pytorch/ 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

# U-Net based autoencoder
# Based this example on Gkamtzir's 3rd model example on github https://github.com/gkamtzir/cnn-image-colorization/blob/main/Network.py

class VideoColorizationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)

        self.t_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.t_conv1_bn = nn.BatchNorm2d(64)
        self.t_conv2 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.t_conv2_bn = nn.BatchNorm2d(32)
        self.t_conv3 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.output = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Implements the forward pass for the given data `x`.
        :param x: The input data.
        :return: The neural network output.
        """
        x_1 = F.relu(self.conv1_bn(self.conv1(x)))
        x_2 = F.relu(self.conv2_bn(self.conv2(x_1)))
        x_3 = F.relu(self.conv3_bn(self.conv3(x_2)))

        x_4 = F.relu(self.t_conv1_bn(self.t_conv1(x_3)))
        x_4 = cat((x_4, x_2), 1)
        x_5 = F.relu(self.t_conv2_bn(self.t_conv2(x_4)))
        x_5 = cat((x_5, x_1), 1)
        x_6 = F.relu(self.t_conv3(x_5))
        x_6 = cat((x_6, x), 1)
        x = self.output(x_6)
        return x


# Initialize and print the model so we can see the architecture 
if __name__ == "__main__":
    model = VideoColorizationCNN()
    print(model)

