# we will do our training in main using our created dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_Dataloader, VideoColorizationDataset
from colorization_cnn2 import VideoColorizationCNN

# number of epochs to train CNN, we can adjust this later
epochs = 10 # was 10
# size of batches, we can modify later but 64 is probably a good start for our larger dataset
batch_size = 64 # was 64

# For testing, limit the number of images to 1, maybe come back to this idea later
# test_limit = 1

# paths for the datasets we created
train_color_folder = './dataset/train/color'
train_grayscale_folder = './dataset/train/grayscale'
test_color_folder = './dataset/test/color'
test_grayscale_folder = './dataset/test/grayscale'


# Create the dataset and print the number of samples
train_dataset = VideoColorizationDataset(train_color_folder, train_grayscale_folder)
print(f"Number of training samples: {len(train_dataset)}")
# Ensure the dataset is not empty
assert len(train_dataset) > 0, "The training dataset is empty." ### 


# dataloaders being created for training and testing
# shuffle true so it doesnt get acclimated to the order
train_loader = create_Dataloader(train_color_folder, train_grayscale_folder, batch_size, shuffle=True)

#test_loader = create_Dataloader(test_color_folder, test_grayscale_folder, batch_size, shuffle=False)

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

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, (L_tensor, ab_tensor) in enumerate(train_loader):
        # Move data to device
        L_tensor = L_tensor.to(device)
        ab_tensor = ab_tensor.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output_ab = model(L_tensor)

        # Compute loss
        loss = criterion(output_ab, ab_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Print loss every 100 batches
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Print average loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {epoch_loss:.4f}')