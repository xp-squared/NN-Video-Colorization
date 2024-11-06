# our CNN that we need to create and train
# Example of CNN : https://www.geeksforgeeks.org/building-a-convolutional-neural-network-using-pytorch/ 
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
            nn.ReLU(),

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

        # DECODER PART
        # https://bluetickconsultants.medium.com/image-and-video-colorization-using-deep-learning-and-opencv-eeec118b58e3
        # upsamples extracted features to produce the AB color channels from LAB
        # Remember L = lightness, A = green-red B = Yellow-Blue
        # this will make it easier for us to adjust , we already have L as the grayscale image so we just need to adjust a and b
        # ConvTranspose2d is deconvolution in pytorch (somewhat)
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,  # takes 256 feature maps from encoder output
                out_channels=128,  # reduce feature map
                kernel_size=3,  # 3x3 kernel
                stride=2, # stride of 2 reverse the downsampling effect, upsample the features by factor of 2
                padding=1,  # adds padding
                output_padding=1 # ensure dimensions are doubled accurately 
            ),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,      # takes 128 feature maps from the previous layer
                out_channels=64,      # reduce feature maps to 64
                kernel_size=3,        
                stride=2,             
                padding=1,            
                output_padding=1      
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,      # takes 64 feature maps from the previous layer
                out_channels=32,      # reduce feature maps to 32
                kernel_size=3,        
                stride=2,             
                padding=1,            
                output_padding=1      
            ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Final layer to produce AB color channels
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=2,      # Outputs 2 channels for `ab` in LAB color space
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # activation function to map output between [-1,1]
            # after normalization a and b will be mapped between [-1,1] so we will try to predict the color within the range
            nn.Tanh()              
        )


    def forward(self, x):
        # example of forward pass from previous assignment
        #out = self.layer1(x)
        #out = self.layer2(out)
        #out = out.reshape(out.size(0), -1)
        #out = self.fc(out)
        #return out
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.final_layer(x)
        return x


# Initialize and print the model so we can see the architecture 
if __name__ == "__main__":
    model = VideoColorizationCNN()
    print(model)

