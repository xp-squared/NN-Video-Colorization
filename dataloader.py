# feed grayscale and color objects to our model
# create a class to do this
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# we do not need to use transforms as when extracting the frames I made each one 224x224

import os
import torch
import numpy as np
import cv2
from PIL import Image

# create dataloader by inheriting from Dataset class in pytorch, good dataloader example
# https://stackoverflow.com/questions/65138643/examples-or-explanations-of-pytorch-dataloaders
class VideoColorizationDataset(torch.utils.data.Dataset):
    def __init__(self,colorFolder,grayScaleFolder):
        self.colorFolder = colorFolder
        self.grayScaleFolder = grayScaleFolder
        # we want to sort the files so we can have consistent pairing when we are doing training with groundtruths
        self.colorImages = sorted(os.listdir(colorFolder))
        self.grayScaleImages = sorted(os.listdir(grayScaleFolder))


    # this will let us return the color and grayscale pair at the given index
    def __getitem__(self, index):
        colorImagePath = os.path.join(self.colorFolder, self.colorImages[index])
        grayscaleImagePath = os.path.join(self.grayScaleFolder, self.grayScaleImages[index])

        # now get the actual images for each and convert to numpy arrays
        # I added convert statements just to make sure they have the correct channels
        colorImage = Image.open(colorImagePath).convert("RGB")
        grayscaleImage = Image.open(grayscaleImagePath).convert("L")

        # converting to numpy arrays, we will do this so we can do operations on the arrays when they are in the LAB space instead of RGB
        # we want to normalize the a b channels between -1 and 1 for our tanh functions 
        colorImage_np = np.array(colorImage)
        grayscaleImage_np = np.array(grayscaleImage)

        # now we want to convert our color image to LAB space
        # Convert color image to LAB color space
        # COLOR IMAGE BECOMES LAB_IMAGE
        LAB_image = cv2.cvtColor(colorImage_np, cv2.COLOR_RGB2LAB)


        #######################################################################
        # Extract 'ab' channels and normalize to [-1, 1]
        ab = (LAB_image[:, :, 1:] - 128) / 128.0  # Shape: [H, W, 2]

        # Normalize grayscale image to [0, 1]
        L = grayscaleImage_np / 255.0  # Shape: [H, W]

        # convert to tensors
        # https://pytorch.org/docs/stable/generated/torch.from_numpy.html
        # make the tensor from the grayscale image
        L_tensor = torch.from_numpy(L).unsqueeze(0).float()          # Shape: [1, H, W]
        # making the tensor from the LAB IMAGE ARRAY
        ab_tensor = torch.from_numpy(ab.transpose((2, 0, 1))).float()  # Shape: [2, H, W]

        return L_tensor, ab_tensor
        ########################################################################3

    # returns length of the dataset
    def __len__(self):
        return len(self.colorImages)

# for now batchsize will be 64 since we have a larger dataset
# https://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model
def create_Dataloader(colorFolder,grayScaleFolder,batchsize, shuffle):
    dataset = VideoColorizationDataset(colorFolder, grayScaleFolder)
    dataloader = torch.utils.data.DataLoader(dataset, batchsize, shuffle=shuffle)
    return dataloader
