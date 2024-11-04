# our CNN that we need to create and train
# Example of CNN : https://www.geeksforgeeks.org/building-a-convolutional-neural-network-using-pytorch/ 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class VideoColorizationCNN(nn.module):
    def __init__(self):
        super(VideoColorizationCNN, self).__init__() # calling constructor of parent class nn.module 

        # we are gonna use sequential to make it more readable : https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # sequential keeps passing output to the next line as input, it will go in order

        # Convolutional layer 1
        self.layer1 = nn.Sequential(
            # first convolutional layer
            # this layer will extract basic edges from the image
            nn.Conv2d(
            in_channels=1,  # grayscale input image, 1 channel
            out_channels=64,  # 64 filters
            kernel_size=3,  # size of the convolutional filter so 3 x 3
            stride=1,  # moving filter 1 pixel at a time
            padding=1  # adds padding to maintain original size, remember conv have problem on edges of images
            ),

            # Batch normalization ensures that the values coming out of each convolutional layer are standardized, 
            # making it easier for the network to learn effectively.
            # IMOW: Batch Normalization fine tines output of convolutional layers so that each batch of data has a normalized mean and variance
            # this helps the network learn quickly and efficiently, mean close to 0 and variance close to 1 across batch of data
            nn.BatchNorm2d(64), # normalizes output of 64 layers

            # activation function for introducing non-linearity to model
            # rectified linear unit (ReLU) activation function
            # We need activation functions to simulate real world data as it is not lienar, it has complex patterns it needs to learn
            # The ReLU function operates by outputting the input directly if it is positive; otherwise, it outputs zero.
            nn.ReLu(),

            # pooling layer to keep the most significant features, as well reduces the spatial dimensions by half for the image
            # takes most important information (MAX)
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # 2 x 2 moving 2 pixels each time
        
        # Convolutional layer 2
        # going deeper and extracting more complex features
        self.layer2 = nn.Sequential(
            nn.Conv2d(
            in_channels=64,  # 64 feature maps from layer 1
            out_channels=128,  # 128 filters, produces 128 feature maps
            kernel_size=3,
            stride=1,  
            padding=1  
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Final convolutional layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(
            in_channels=128,  # 128 feature maps from layer 2
            out_channels=256,  # 256 filters, produces 256 feature maps
            kernel_size=3,  
            stride=1,  
            padding=1  
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # I think more than 3 layers would be overkill, 2 might even work fine but lets start with 3 for now
        

    def forward(self, x):
        # example of forward pass from previous assignment
        #out = self.layer1(x)
        #out = self.layer2(out)
        #out = out.reshape(out.size(0), -1)
        #out = self.fc(out)
        #return out
        x = self.layer1
        x = self.layer2
        x = self.layer3
        # ...

# Loss and optimizer, I am going to use the same one from CNN class example
# Cross-entropy loss is a commonly used metric in machine learning, 
# particularly for classification tasks, that measures the difference between 
# the probability distribution predicted by a model and the true probability distribution of the data
criterion = nn.CrossEntropyLoss()
# maybe change LR later, better to keep it smaller for now
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

# step 4 on geeks for geeks is very important for backward propagation