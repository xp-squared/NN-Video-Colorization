import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torch import no_grad, cat
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import numpy as np
from colorization_models import Network1, Network3, Network6
from torchvision import transforms
import cv2
import os

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


def reassemble_color_video(coloredVideoFolder, inputColoredFrames, ogVideo): 
    video = cv2.VideoCapture(ogVideo)
    # see if video was opened successfully
    if not video.isOpened():
        print(f"Error could not open video: {ogVideo}")
        return
    fps = video.get(cv2.CAP_PROP_FPS) # getting the frames per second
    # width and height of each video frame, for now we will keep it 400x400 we could implement it to make it the original frame size later
    frame_width = 400 
    frame_height = 400
    # did this so we can make sure the output video has the same FPS
    video.release()

    outputVidPath = os.path.join(coloredVideoFolder, 'colored_output_video.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # now we will reassemble our colored frames into a video
    new_video = cv2.VideoWriter(outputVidPath, fourcc, fps, (frame_width, frame_height))

    for frame in os.listdir(inputColoredFrames):
        path = os.path.join(inputColoredFrames, frame)
        frame = cv2.imread(path)
        new_video.write(frame)

    new_video.release()
    cv2.destroyAllWindows()

    print(f"Video saved to {outputVidPath}")
    

#####################################################################################################
def color_frames(image_path, framecount, outputFolder):
    choice = 6 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TO CHANGE NEURAL NETWORK CHOICE
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
    #colorized_rgb = lab2rgb(colorized_lab) ###############
    colorized_lab = np.clip(colorized_lab, 0, 100)  # L is typically 0-100, AB can be outside this range
    colorized_rgb = lab2rgb(colorized_lab)

    # Convert back to BGR for OpenCV saving
    colorized_bgr = cv2.cvtColor((colorized_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # save the colored frame to the folder
    colored_frame_fileName = os.path.join(outputFolder, f"frame_{framecount:05d}.jpg")
    cv2.imwrite(colored_frame_fileName, colorized_bgr)
    ##################################################################################################### MAYBE PUT THIS IN COLORIZE SINGULAR TOO



def extract_frames(video_path, stored_frames, resize_dim=(400, 400)):

    # video to read from
    video = cv2.VideoCapture(video_path)

    # error opening video
    if not video.isOpened():
        print(f"Error could not open video: {video_path}")
        return
    
    # keeping track of how many frames
    frameCount = 0

    # read first frame from video
    # will check if success was success by the loop continuing
    success, frame = video.read()

    # loop till no more frames
    while success: # SUCCESS
        if resize_dim:
            frame = cv2.resize(frame, resize_dim)

        # convert frame to grayscale just to make sure, this may be unneeded tho
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # save the grayscale frame for now, but we are gonna send the current frame to the colorize function
        grayscale_frame_fileName = os.path.join(stored_frames, f"frame_{frameCount:05d}.jpg")
        cv2.imwrite(grayscale_frame_fileName, frame)

        #colored_frame = color_frames(gray_frame, stored_frames, frameCount, cnn)

        success, frame = video.read()
        frameCount += 1




if __name__ == '__main__':
    # folder where our test videos will be
    # catVid_1_1000_1020
    # catVid_1_1339_1349
    video_path = 'TestVideos/City.mp4'

    # folder that will hold the temporarily extracted frames
    # as we color them we will save the frame to its grayscale version
    # as we reassemble the video we will delete the frames
    framesFromVideo = 'CurrentExtractedFrames'
    coloredFrameFolder= 'CurrentColoredFrames'
    videoFolder = 'ColoredVideo'

    extract_frames(video_path,framesFromVideo) # getting the extracted frames from the video

    # now we want to take the current extracted frames and color each one, saving it to the same name

    image_extensions = ('.jpg')
    # loop through all files in the folder
    framecount = 0
    for filename in os.listdir(framesFromVideo):
        # Check if the file is an image based on its extension
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(framesFromVideo, filename)
            color_frames(image_path, framecount, coloredFrameFolder)
            framecount +=1

    # now we want to reassemble the colored frames into a video and as well delete the frames that we had made
    reassemble_color_video(videoFolder, coloredFrameFolder, video_path)


    for frame in os.listdir(framesFromVideo):
        frame_path = os.path.join(framesFromVideo, frame)
        os.remove(frame_path)

    for frame in os.listdir(coloredFrameFolder):
        frame_path = os.path.join(coloredFrameFolder, frame)
        os.remove(frame_path)




