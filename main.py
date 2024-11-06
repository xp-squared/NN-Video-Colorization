# we will do our training in main using our created dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_Dataloader
from colorization_cnn import VideoColorizationCNN

# number of epochs to train CNN, we can adjust this later
epochs = 10
# size of batches, we can modify later but 64 is probably a good start for our larger dataset
batch_size = 64

# paths for the datasets we created
train_color_folder = '.dataset/train/color'
train_grayscale_folder = '.dataset/train/grayscale'
test_color_folder = '.dataset/test/color'
test_grayscale_folder = '.dataset/test/grayscale'

# dataloaders being created for training and testing
# shuffle true so it doesnt get acclimated to the order
train_dataset = create_Dataloader(train_color_folder,train_grayscale_folder,batch_size, shuffle=True)
test_dataset = create_Dataloader(test_color_folder,test_grayscale_folder,batch_size, shuffle= False)

# initialize our model 
model = VideoColorizationCNN()
# if true uses gpu, else uses cpu, this enhances performance and was used in a previous CNN assignment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer, I am going to use the same one from CNN class example
# Mean Squared Error
criterion = nn.MSELoss()
# maybe change LR later, better to keep it smaller for now
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

# training loop
for epoch in range(epochs):

# step 4 on geeks for geeks is very important for backward propagation