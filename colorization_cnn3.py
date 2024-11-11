# our CNN that we need to create and train
# Example of CNN : https://www.geeksforgeeks.org/building-a-convolutional-neural-network-using-pytorch/ 
import torch
import torch.nn as nn

class VideoColorizationCNN(nn.Module):
    def __init__(self):
        super(VideoColorizationCNN, self).__init__() # calling constructor of parent class nn.module 

        # we are gonna use sequential to make it more readable : https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # sequential keeps passing output to the next line as input, it will go in order

        # Encoder and Decoder: The encoder takes the input data and compresses it into a lower-dimensional representation called the 
        # latent space. The decoder then reconstructs the input data from the latent space representation.

        # ENCODER PART OF CODE
        # extracts features from grayscale image
        # Convolutional layer 1
        self.enc1 = nn.Sequential(
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
            nn.ReLU(),

            # pooling layer to keep the most significant features, as well reduces the spatial dimensions by half for the image
            # takes most important information (MAX)
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # 2 x 2 moving 2 pixels each time
        
        # Convolutional layer 2
        # going deeper and extracting more complex features
        self.enc2 = nn.Sequential(
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
        self.enc3 = nn.Sequential(
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
        
        # adding another layer
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # DECODER PART
        # https://bluetickconsultants.medium.com/image-and-video-colorization-using-deep-learning-and-opencv-eeec118b58e3
        # github from this project: https://github.com/bluetickconsultants/image_video_colorization/blob/main/TF%20.ipynb
        # below is a good example of how Transpose2d works
        # https://indico.cern.ch/event/996880/contributions/4188468/attachments/2193001/3706891/ChiakiYanagisawa_20210219_Conv2d_and_ConvTransposed2d.pdf
        # upsamples extracted features to produce the AB color channels from LAB
        # Remember L = lightness, A = green-red B = Yellow-Blue
        # this will make it easier for us to adjust , we already have L as the grayscale image so we just need to adjust a and b
        # ConvTranspose2d is deconvolution in pytorch (somewhat)

        # RESEARCH SKIP CONNECTIONS MORE, THAT IS WHAT WE ARE USING
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # 512 = 256 + 256 (skip)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,      # takes 128 feature maps from the previous layer
                out_channels=64,      # reduce feature maps to 64
                kernel_size=4,        
                stride=2,             
                padding=1,               
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128 = 64 + 64 (from skip)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Final layer to produce AB color channels
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh()  # Output range [-1, 1]
        )


    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Decoder with skip connections
        d4 = self.dec4(e4)
        d4_cat = torch.cat([d4, e3], dim=1)  # Skip connection from enc3
        
        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, e2], dim=1)  # Skip connection from enc2
        
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e1], dim=1)  # Skip connection from enc1
        
        d1 = self.dec1(d2_cat)
        
        out = self.final(d1)
        return out


# Initialize and print the model so we can see the architecture 
if __name__ == "__main__":
    model = VideoColorizationCNN()
    print(model)

