import torch.nn as nn
import torch.nn.functional as F
from torch import cat


# model 1 that I have created, a simple encoder and decoder architecture 
class Network1(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder layers
        self.layer1 = nn.Sequential(
            # first convolutional layer
            # this layer will extract basic edges from the image
            nn.Conv2d(
                in_channels=1,  # grayscale input image, 1 channel
                out_channels=32,  # 32 filters
                kernel_size=4,  # size of the convolutional filter so 4 x 4
                stride=2,  # moving filter 2 pixel at a time
                padding=1  # adds padding to maintain original size, remember conv have problem on edges of images
            ),
            # Batch normalization ensures that the values coming out of each convolutional layer are standardized, 
            # making it easier for the network to learn effectively.
            # IMOW: Batch Normalization fine tines output of convolutional layers so that each batch of data has a normalized mean and variance
            # this helps the network learn quickly and efficiently, mean close to 0 and variance close to 1 across batch of data
            nn.BatchNorm2d(32) # normalizes output of 32 layers

            # do relu in the forward part instead this time
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )


        # decoder layer
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)
        )


    def forward(self, x):
        # doing the activation function after each layer using RELU
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x


# more detailed model to test images with from gkamtzir on github https://github.com/gkamtzir, model 3 from his project
class Network3(nn.Module):
    def __init__(self):
        """
        Initializes each part of the convolutional neural network.
        """
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
    

# more detailed model to test images with from gkamtzir on github https://github.com/gkamtzir, model 6 from his project
class Network6(nn.Module):
    def __init__(self):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)

        # Dilation layers.
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv6_bn = nn.BatchNorm2d(256)

        self.t_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.t_conv1_bn = nn.BatchNorm2d(128)
        self.t_conv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.t_conv2_bn = nn.BatchNorm2d(64)
        self.t_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.t_conv3_bn = nn.BatchNorm2d(32)
        self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

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
        x_4 = F.relu(self.conv4_bn(self.conv4(x_3)))

        # Dilation layers.
        x_5 = F.relu(self.conv5_bn(self.conv5(x_4)))
        x_5_d = F.relu(self.conv6_bn(self.conv6(x_5)))

        x_6 = F.relu(self.t_conv1_bn(self.t_conv1(x_5_d)))
        x_6 = cat((x_6, x_3), 1)
        x_7 = F.relu(self.t_conv2_bn(self.t_conv2(x_6)))
        x_7 = cat((x_7, x_2), 1)
        x_8 = F.relu(self.t_conv3_bn(self.t_conv3(x_7)))
        x_8 = cat((x_8, x_1), 1)
        x_9 = F.relu(self.t_conv4(x_8))
        x_9 = cat((x_9, x), 1)
        x = self.output(x_9)
        return x

# print the model so we can see the architecture 
if __name__ == "__main__":
    model = Network1()
    print(model)
    print()

