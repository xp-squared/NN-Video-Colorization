import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from colorization_models import Network1, Network3, Network6
from torchvision import transforms
import cv2
import os 

def current_Network(choice):
    if choice == 1:
        return Network1(), 'Models\colorization_model1_OG.pth'  # original network, no step training
    elif choice == 2:
        return Network1(), 'Models\colorization_model1_epoch2000.pth' # new network 1 with deeper training, trained by omen GPU
    elif choice == 3:
        return Network3(), 'Models\colorization_model3_OG.pth' 
    elif choice == 6:
        return Network6(), 'Models\colorization_model6_OG.pth' # original network no step training, EHHHHH not that great
    elif choice == 7:
        return Network6(), 'Models\colorization_model6.pth'  # new network 6 with deeper training, trained on Asus
    else:
        raise ValueError("Invalid choice. Please select 1, 2, 3, 6, or 7")

def test_single_image(image_path):
    choice = 2 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TO CHANGE NEURAL NETWORK CHOICE
    cnn, model_file = current_Network(choice) # based on choice, we will grab the network and load its path from when we trained it
    cnn.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'))) # either use cpu or gpu depending on device
    cnn.eval() # evaluation mode for the model

    transform = transforms.Compose([
        transforms.Resize((400, 400)),  # resize image to 400x400 pixels
    ])

    # read the image in color (not grayscale)
    image = cv2.imread(image_path)

    # convert from BGR to RGB because cv2 reads in BGR 
    # BGR is a representation of an image where the order of the color channels is different from RGB.
    # The main difference between BGR and RGB is the order in which the color channels are specified for each pixel. In BGR, the order is blue, green, and red
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # first we convert the image to a tensor
    # then we rearrange the tensor from H,W,C to C,H,W to match pytorch format
    # we then apply the transformation making the image 400x400
    # makes tensor back to HWC
    # converts back to numpy array
    # This whole line makes sure that the image is the required dimensions and we make sure it is in correct format for the next steps
    image = transform(torch.from_numpy(image).permute(2, 0, 1)).permute(1, 2, 0).numpy() 

    
    image_lab = rgb2lab(image).astype('float32')  # Convert the RBG image to LAB using the function, in float datatype

    # convert the lab image to a tensor and change to C,H,W to match pytorch format
    # C = 3, L A and B channels
    image_lab_tensor = torch.from_numpy(image_lab).permute(2, 0, 1) 

    # Extract L channel
    # we retain the same dimensions but we make sure to grab the first channel which is the L channel making the grayscale image!
    # same format
    gray_image = image_lab_tensor[0:1, :, :]  # Shape: [1, H, W]

    # Run the model
    with torch.no_grad(): # no gradient computations when you are testing
        # adds a new dimension at position 0 to represent the batch size so we go from 1,H,W to 1,1,H,W
        # we then pass the gray image through the neural network  to predict a and b, 
        # we add the new dimension cos neural network expects another dimension
        # the result will be 1,2,H,W
        output_ab = cnn(gray_image.unsqueeze(0))  

        # remove that extra dimension of batch so we now just have size of 2,H,W
        # we do this so we can concatenate the grayscale image channel with the predicted a and b channel from this
        output_ab = output_ab.squeeze(0) 

    # combining L channel with predicted AB channels
    # first we concatenate the gray and output together using dimension zero as that is where the channels position are in the tensor
    # we theen rearrance the tensor from C,H,W to H,W,C to turn back to a numpy array
    colorized_lab = torch.cat((gray_image, output_ab), dim=0).permute(1, 2, 0).numpy()  

    # converts lab predicted image back to rbg so we can visualize it
    colorized_rgb = lab2rgb(colorized_lab)

    # Convert back to BGR for OpenCV saving so we can save the images to our folder, 
    colorized_bgr = cv2.cvtColor((colorized_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # get file name 
    # fixed weird naming convention from earlier too
    filename = os.path.basename(image_path)

    # get ready to save colored image 
    colored_frame_fileName = os.path.join('ColoredImages', f"Colored_MC{choice}_{filename}")

    # saving image
    cv2.imwrite(colored_frame_fileName, colorized_bgr)

    # Display the images
    # figure for plotting 12x6 
    plt.figure(figsize=(12, 6))

    # Display the grayscale input
    plt.subplot(1, 2, 1) # 1 row 2 col, first plot of grayscale image
    plt.imshow(gray_image.squeeze(0).numpy(), cmap='gray') # change array from 1,H,W to H,W by squeezing first index and convert to a numpy array. Show grayscale image
    plt.title('Grayscale Input')
    plt.axis('off') # hides ticks

    # Display the colorized output
    plt.subplot(1, 2, 2) # 1 row 2 col plot prediction! 
    plt.imshow(colorized_rgb)
    plt.title('Colorized Output')
    plt.axis('off')

    plt.show() # show the output! 


if __name__ == '__main__':

    image_path = 'TestImages\Butterfly.jpg'  # Replace with the path to your image

    test_single_image(image_path)
