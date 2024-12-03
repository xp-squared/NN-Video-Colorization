import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torch import from_numpy, save
from skimage.color import rgb2lab
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from colorization_models import Network1, Network3, Network6

# using GPU from other computer to train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ###
print(f"Using device: {device}") ###

def current_Network(choice):
    if choice == 1:
        return Network1(), 'Models\colorization_model1_OG.pth'  # original network, no step training
    elif choice == 2:
        return Network1(), 'Models\colorization_model1.pth' # new network 1 with deeper training 
    elif choice == 3:
        return Network3(), 'Models\colorization_model3_OG.pth' 
    elif choice == 6:
        return Network6(), 'Models\colorization_model6_OG.pth' # original network no step training
    elif choice == 7:
        return Network6(), 'Models\colorization_model6.pth'  # new network 6 with deeper training 
    else:
        raise ValueError("Invalid choice. Please select 1, 2, 3, 6, or 7")

# wanted to make a same class for dataloading like gkamtzir on github so we can load the data https://github.com/gkamtzir, directly followed from him but added more comments.
class ImageDataset(Dataset):
    def __init__(self, color_dir, gray_dir = None, transform = None, target_transform = None): 
        # initialize dataset with the names, color directory, the gray directory set to none as a standard as well as transform and target_transform being set to none if no parameters
        self.names = os.listdir(color_dir)[:5000]# this was used to have less steps for training
        self.color_dir = color_dir
        self.gray_dir = gray_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # able to return length 
        return len(self.names)

    def __getitem__(self, index):
        # getting item from the directories, never used within my code
        if self.gray_dir is not None:
            gray_path = os.path.join(self.gray_dir, self.names[index])
            gray_image = read_image(gray_path, ImageReadMode.GRAY)

            color_path = os.path.join(self.color_dir, self.names[index])
            color_image = read_image(color_path)
        else:
            color_path = os.path.join(self.color_dir, self.names[index])
            image = from_numpy(rgb2lab(read_image(color_path).permute(1, 2, 0))).permute(2, 0, 1)

            # The color image consists of the 'a' and 'b' parts of the LAB format.
            color_image = image[1:, :, :]
            # The gray image consists of the L part of the LAB format.
            gray_image = image[0, :, :].unsqueeze(0)

        if self.transform:
            gray_image = self.transform(gray_image)
        if self.target_transform:
            color_image = self.target_transform(color_image)

        return gray_image, color_image
    


# Training function, followed gkamtzir once again but branched off on my own design, just want to credit him for inspiration
def train_model(color_dir, gray_dir=None, epochs=1000, learning_rate=0.001, batch_size=32):

    # loading the dataset and creating a dataloader based on the loaded dataset, we determine our batch size being 32 and we shuffle the data to train
    training_data = ImageDataset(color_dir=color_dir, gray_dir=gray_dir)
    train_data_loader = DataLoader(
        training_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=6,  # Use multiple CPU cores for data loading, VARIES BY COMPUTER
        pin_memory=True  # Faster data transfer to GPU
    )
    # create the network and change choice to use either network 1, 2, 3, 6, 7
    choice = 6 # new model 1
    cnn, modelpath = current_Network(choice) # not gonna use modelpath in this code so no worries

    # moving the model to the device
    cnn.to(device) ###

    criterion = nn.MSELoss() # using mean squared error loss, we measure the predicted color vals with the ground truth to get the loss
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    # scheduler to have an adaptive learning rate
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    print(f"Number of parameters: {sum(p.numel() for p in cnn.parameters())}")
    for epoch in range(epochs):
        epoch_running_loss = 0
        for i, data in enumerate(train_data_loader, 0):
            gray, color = data
            gray = gray.float().to(device)    # Move gray images to device ###
            color = color.float().to(device)  # Move color images to device ### 

            outputs = cnn(gray)

            optimizer.zero_grad()
            loss = criterion(outputs, color)
            loss.backward()
            optimizer.step()

            epoch_running_loss += loss.item()
            if (i + 1) % 1 == 0:  # This will print every step
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_data_loader)}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {epoch_running_loss / len(train_data_loader):.4f}')
        
        if (epoch + 1) % 25 == 0:
            checkpoint = f'model{choice}_checkpoint_epoch{epoch+1}.pth'
            torch.save(cnn.state_dict(), checkpoint)
            save(cnn.state_dict(), modelpath)
            print(f"Model checkpoint saved at epoch {epoch + 1} as {checkpoint}")

        scheduler.step()

        # print the new learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f'Learning rate adjusted to: {current_lr:.6f}')
    print("Finished Training")

    # Save the trained model
    save(cnn.state_dict(), modelpath)
    print("Model saved as", modelpath)

# Example usage
if __name__ == '__main__':
    color_dir = './data/dataset/train_color'  # Replace with your color images directory
    gray_dir = None  # Set to None if using LAB conversion within the dataset
    epochs = 1000
    learning_rate = 0.001
    batch_size = 32 # larger batch size so we can speed up the training process, was 32 but steps were too slow

    train_model(color_dir=color_dir, gray_dir=gray_dir, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

    # for our og models, batch size was 32 and [:4] was uncommented within the image dataset class
