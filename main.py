import torch
import torch.nn as nn
import torch.optim as optim
from colorization_cnn import VideoColorizationCNN
import torchvision
import numpy as np
import torchvision.transforms as transforms
from skimage import color


# number of epochs to train CNN, we can adjust this later, baseline: 10
epochs = 10 
# size of batches, we can modify later but 64 is probably a good start for our larger dataset, baseline : 64
batch_size = 64 
learning_rate = .001 # learning rate, baseline: .001


# transformations for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize images to 224x224
    transforms.ToTensor(),          # convert images to tensors
])



# https://pytorch.org/tutorials/recipes/recipes/loading_data_recipe.html
# Using CIFAR-10 Dataset instead of our frame extracted dataset, we will see how frame consitency is later down the line but was getting tough to implement
train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,  # loading the train set
                                             transform=transform, # apply transformation defined above
                                             download=True # download the dataset if not available
                                             )
# num_workers option could be set to 2, from my understanding you would have 2 workers getting data and putting it into ram, setting workers to amount of cores is smart too
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # load dataset
                                            batch_size=batch_size, # number of samples per batch
                                            shuffle=True, # shuffle data!
                                            num_workers=2 # processes for loading data (think pthreads from os class maybe)
                                            )


model = VideoColorizationCNN() # initialize our model 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # if true uses gpu, else uses cpu, this enhances performance and was used in a previous CNN assignment
model = model.to(device)

# Loss and optimizer
criterion = nn.MSELoss() # Mean Squared Error, will calc the difference between predicted and target vals
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # LR BASELINE: .001

# Training loop
print("Starting training...")
for epoch in range(epochs):
    # we will convert the images used from training directly to grayscale while down here
    model.train()                # we will set model to training mode, this comes from nn.module class, the class we inherited methods from for our CNN, will affect some layers
    running_loss = 0.0           # calculate loss over each epoch

    # iterating over batches of the data
    for batch, (images, NA) in enumerate(train_loader): # we do not need to use NA as it is the labels in CIFAR-10
        images = images.to(device)

        # we want to now convert the images from RGB to LAB and retrieve the AB from our model
        # list to store the LAB images
        LAB_images = []
        for img in images:  # iterate over images and convert them
            img_np = np.array(img) # converting image to numpy array, it has shape (height, width, 3) (3 channels for color), gray scale image only has (height, width)

            # do we have to do the transpose line?

            lab = color.rgb2lab(img_np) # converting image to LAB colorspace
            # remember we are converting to the lab channel
            # L lightness : ranges from 0 to 100, we are gonna use the regular L channel without AB for grayscale
            # we need to normalize the a and b channel to get between -1 and 1
            # A Green Red : ranges from -128 to 127
            # B Blue Yellow : ranges from -128 to 127

            LAB_images.append(lab)

        # https://www.geeksforgeeks.org/numpy-stack-in-python/ , basically makes 2d array of all our images by row
        LAB_images = np.stack(LAB_images, axis=0)

        # now we are going to split LAB images into L and AB channels
        L = LAB_images[:,:,:,0] # L channel, the shape : (batchsize, H, W) 1 Single channel
        ab = LAB_images[:,:,:,1:] # ab channels, shape is (batchsize, h, w, 2) 2 channels 

        # normalizing the l channel, remenber its between 0 and 100 
        L = L / 100.0 # this makes it between [0,1]

        # normalizing the ab channels between [-1,1]
        ab = ab / 128.0

        # conver L and ab channels (numpy) back to tensors
        # need to add back a dimension to L since L and ab do not have matching dimensions ##################################################
        L_tensor = torch.from_numpy(L).unsqueeze(1).to(device) # shape (batch_size, 1, h, w)
        # need to move around shape for ab tensor to be (batch_size, 2, H, W) like L tensor
        ab_tensor = torch.from_numpy(ab).permute(0,3,1,2).to(device)




        

print("Finished training...")