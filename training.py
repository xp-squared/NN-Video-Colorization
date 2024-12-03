import os # use to manipulate file and directories
import torch # used for the nn and other operations
from torch.utils.data import Dataset, DataLoader # class for making a dataset, using the class that was made and dataloader
from torchvision.io import read_image, ImageReadMode # being able to to read image and specific color mode
from torch import from_numpy, save # converting numpy array to tensors and saving the models
from skimage.color import rgb2lab # convert rgb to lab space for getting a and b vals
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler # scheduler to adjust the learning rate
import torch.nn as nn # importing part of nn modules
from colorization_models import Network1, Network3, Network6 # importing the networks that we created 

# using GPU from other computer to train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # we check if there is a gpu available to use if not we just use the cpu
print(f"Using device: {device}") # will tell us what device is being used

def current_Network(choice):
    if choice == 1:
        return Network1(), 'Models\colorization_model1_OG.pth'  # original network, no step training
    elif choice == 2:
        return Network1(), 'Models\colorization_model1.pth' # new network 1 with deeper training 
    elif choice == 3:
        return Network3(), 'Models\colorization_model3_OG.pth'  # intermediate network, kind of ignored and just focused on 1 and 6
    elif choice == 6:
        return Network6(), 'Models\colorization_model6_OG.pth' # original network no step training
    elif choice == 7:
        return Network6(), 'Models\colorization_model6.pth'  # new network 6 with deeper training 
    else:
        raise ValueError("Invalid choice. Please select 1, 2, 3, 6, or 7")

# wanted to make a same class for dataloading like gkamtzir on github so we can load the data https://github.com/gkamtzir, directly followed from him but added more comments and changed small things.
class ImageDataset(Dataset): # gray dir will be none if not specified as well as transformations
    def __init__(self, color_dir, gray_dir = None, transform = None, target_transform = None): 
        # initialize dataset with the names, color directory, the gray directory set to none as a standard as well as transform and target_transform being set to none if no parameters
        self.names = os.listdir(color_dir)  [:150] # this was used to have less steps for training, can be changed probably up to 5000 or just remove
        self.color_dir = color_dir
        self.gray_dir = gray_dir 
        self.transform = transform # no need for transformation all images are 400x400
        self.target_transform = target_transform # no need

    def __len__(self):
        # able to return length 
        return len(self.names)

    def __getitem__(self, index):
        # getting item from the directories, has to be implemented from the dataset class 
        if self.gray_dir is not None:
            gray_path = os.path.join(self.gray_dir, self.names[index])
            gray_image = read_image(gray_path, ImageReadMode.GRAY)

            color_path = os.path.join(self.color_dir, self.names[index])
            color_image = read_image(color_path)
        else:
            color_path = os.path.join(self.color_dir, self.names[index])
            # reads the imaeg and changes from c,h,w to h,w,c, converts to lab and then to a tensor (md array containting elements of the data), permute tensor back to c,h,w
            image = from_numpy(rgb2lab(read_image(color_path).permute(1, 2, 0))).permute(2, 0, 1)

            # The color image consists of the 'a' and 'b' parts of the LAB format.
            color_image = image[1:, :, :] 
            # The gray image consists of the `L` part of the LAB format.
            gray_image = image[0, :, :].unsqueeze(0) # just the l part so we have our gray image

        if self.transform:
            gray_image = self.transform(gray_image)
        if self.target_transform:
            color_image = self.target_transform(color_image)

        return gray_image, color_image
    


# Training function, followed gkamtzir once again but branched off on my own design, just want to credit him for inspiration
def train_model(color_dir, gray_dir, epochs, learning_rate, batch_size):
    # parameters the function takes
    # color_dir, the directory containing the color image
    # directory containing grayscale images but we actually just convert them here
    # epochs, the nummber of times we run the network
    # batch size which shows the number of samples processed before model is updated

    # loading the dataset and creating a dataloader based on the loaded dataset, we determine our batch size being 32 and we shuffle the data to train
    # we create an instance of the image dataset class with the directories
    training_data = ImageDataset(color_dir, gray_dir)
    # lets us use the dataset and we can shuffle and have batches 
    train_data_loader = DataLoader(training_data, batch_size, shuffle=True)

    # create the network and change choice to use either network 1, 2, 3, 6, 7
    choice = 7 #############################################################################################################################################################
    cnn, modelpath = current_Network(choice) # load network choice and get its modelpath that we will save
    # moving the model to the device either cpu or gpu
    cnn.to(device) 

    criterion = nn.MSELoss() # using mean squared error loss, we measure the predicted color vals with the ground truth to get the loss
    optimizer = optim.Adam(cnn.parameters(), learning_rate) # initializing the optimizer

    # scheduler to have an adaptive learning rate
    # reduces the learning rate every 50 epochs, maybe adjust to 100 or 150 
    # Gamma rate to decrease learning by doing lr = lr * .5
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    print(f"Number of parameters: {sum(p.numel() for p in cnn.parameters())}")
    # loops over the amount of epochs giving
    for epoch in range(epochs): 
        epoch_running_loss = 0 # used to calculate the average loss each epoch
        for i, data in enumerate(train_data_loader, 0):
            # take data from getItem function, we get grayscale tensors, and a/b tensors for color
            gray, color = data # retrieves the gray and color image

            # gray has a shape of (32, 1, 400, 400). 32 is batch size, 1 channel being L channel and 400 x 400 image
            # color has a shape of (32, 2, 400, 400). 32 batch size, 2 channels a and b

            gray = gray.float().to(device)    # convert the gray images to a float tensor and moves to cpu or gpu, transforms pixel vals to floats
            color = color.float().to(device)  # same above but with color image

            # Tensor is a md array that stores data
            # above we get the pixel tensors for gray and color

            outputs = cnn(gray) # forward pass thru network to get the predictions of a and b, we insert the gray tensor through
            # get output of shape (32, 2, 400, 400)

            optimizer.zero_grad() # clears gradient from previous iterations
            loss = criterion(outputs, color) # computes loss between prediction and actual color channels
            loss.backward() # backpropagation to learn from mistake
            optimizer.step() # this updates paramaters based on the loss

            epoch_running_loss += loss.item() # accumulating loss for epoch
            if (i + 1) % 1 == 0:  # show the loss for each step (batch)
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_data_loader)}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {epoch_running_loss / len(train_data_loader):.4f}') # finishing epoch and showing average loss

        scheduler.step() # update learning rate according to scheduler

        # gets the current learning rate now that it has been adjusted
        current_lr = scheduler.get_last_lr()[0]
        print(f'Learning rate adjusted to: {current_lr:.6f}')
    print("Finished Training")

    # save the trained model
    save(cnn.state_dict(), modelpath) # saves model as path defined when chossing the network to use
    print("Model saved as", modelpath)

# Example usage
if __name__ == '__main__':
    color_dir = './data/dataset/train_color'  # Replace with your color images directory
    batch_size = 32 # larger batch size so we can speed up the training process, was 32 but steps were too slow

    train_model(color_dir=color_dir, gray_dir=None, epochs=1000, learning_rate=.001, batch_size=32)

    # for our og models, batch size was 32 and [:4] was uncommented within the image dataset class
