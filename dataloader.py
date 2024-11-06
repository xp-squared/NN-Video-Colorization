# feed grayscale and color objects to our model
# create a class to do this
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# we do not need to use transforms as when extracting the frames I made each one 224x224

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Dataloader(Dataset):
    def __init__(self,colorFolder,grayScaleFolder):
        self.colorFolder = colorFolder
        self.grayScaleFolder = grayScaleFolder
        # we want to sort the files so we can have consistent pairing when we are doing training with groundtruths
        self.colorImages = sorted(os.listdir(colorFolder))
        self.grayScaleImages = sorted(os.listdir(grayScaleFolder))



def create_Dataloader():
