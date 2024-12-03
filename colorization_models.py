import torch.nn as nn
import torch.nn.functional as F
from torch import cat


# model 1 that I have created, a simple encoder and decoder architecture 
# the input is a grayscale image which has 1 channel
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
            nn.BatchNorm2d(32) # normalizes output of 32 layers, making sure data has mean of 0 and std of 1

            # do relu in the forward part instead this time
        )

        # increases feature maps from 32 to 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )


        # decoder layer
        self.layer3 = nn.Sequential(
            # pytorchs version of deconvolution
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32)
        )

        self.layer4 = nn.Sequential(
            # reduces feature maps from 32 to 2 representing the a and b color channels
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)
        )

        # data flow of network

        # layer 1 
        # input : (32, 1, 400, 400), 32 image batch, 1 channel which is grayscale, size 400 x 400
        #  output: (32, 32, 200, 200), 32 image batch still, 32 feature maps, reduce sized 200x200

        # layer 2:
        # input : (32, 32, 200, 200), 32 image batch , 32 feature maps, reduce sized 200x200
        # output : (32, 64, 100, 100), 32 image batch, 64 feature maps, reduce sized 100x100

        # layer 3: deconvolution (transpose)
        # input : (32, 64, 100, 100), 32 image batch, 64 feature maps, sized 100x100
        # output : (32, 32, 200, 200), 32 image batch, 32 feature maps, back sized at 200x200

        # layer 4: 
        # input : (32, 32, 200, 200), 32 image batch, 32 feature maps, back sized at 200x200 
        # output : (32, 2, 400, 400), 32 image batch, 2 output channels representing a and b, sized 400x400

    def forward(self, x):
        # doing the activation function after each layer using RELU
        # we extract features by doing this
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x


# more detailed model to test images with from gkamtzir on github https://github.com/gkamtzir, model 3 from his project
# model that has skip connections
# A skip connection, also known as a shortcut connection, is a link in a neural network that
#  allows information to pass from one layer to another without going through all the layers in between
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
# has 4 convolutional layers, 2 dilation layers, 4 deconvolution layers (transpose)
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

