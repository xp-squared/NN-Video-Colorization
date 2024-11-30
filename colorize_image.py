import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from colorization_models import Network1, Network3, Network6
from torchvision import transforms
import cv2

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

def test_single_image(image_path):
    choice = 1 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TO CHANGE NEURAL NETWORK CHOICE
    cnn, model_file = current_Network(choice)
    cnn.load_state_dict(torch.load(model_file, weights_only=True))
    cnn.eval()

    transform = transforms.Compose([
        transforms.Resize((400, 400)),  # Resize the image to 400x400 pixels
    ])

    # Read the image in color (not grayscale)
    image = cv2.imread(image_path)
    # Convert from BGR to RGB (cv2 reads in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the transformations
    image = transform(torch.from_numpy(image).permute(2, 0, 1)).permute(1, 2, 0).numpy()

    # Load and preprocess the image
    image_lab = rgb2lab(image).astype('float32')  # Convert to LAB color space
    image_lab_tensor = torch.from_numpy(image_lab).permute(2, 0, 1)  # Convert to [C, H, W]

    # Extract L channel
    gray_image = image_lab_tensor[0:1, :, :]  # Shape: [1, H, W]

    # Run the model
    with torch.no_grad():
        output_ab = cnn(gray_image.unsqueeze(0))  # Add batch dimension, Shape: [1, 2, H, W]
        output_ab = output_ab.squeeze(0)  # Remove batch dimension, Shape: [2, H, W]

    # Combine L channel with predicted AB channels
    colorized_lab = torch.cat((gray_image, output_ab), dim=0).permute(1, 2, 0).numpy()  # Shape: [H, W, 3]

    # Convert LAB to RGB
    colorized_rgb = lab2rgb(colorized_lab)

    # Convert back to BGR for OpenCV saving
    colorized_bgr = cv2.cvtColor((colorized_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Display the images
    plt.figure(figsize=(12, 6))

    # Display the grayscale input
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image.squeeze(0).numpy(), cmap='gray')
    plt.title('Grayscale Input')
    plt.axis('off')

    # Display the colorized output
    plt.subplot(1, 2, 2)
    plt.imshow(colorized_rgb)
    plt.title('Colorized Output')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':

    image_path = 'TestImages\Tiger.jpg'  # Replace with the path to your image

    test_single_image(image_path)
