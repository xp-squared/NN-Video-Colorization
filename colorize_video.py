import torch
import torch.nn as nn
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from colorization_models import Network1, Network3, Network6
from torchvision import transforms
import cv2
import os # used to move around files

def current_Network(choice):
    if choice == 1:
        return Network1(), 'Models\colorization_model1_OG.pth'  # original network, no step training
    elif choice == 2:
        return Network1(), 'Models\colorization_model1_epoch2000.pth' # new network 1 with deeper training 
    elif choice == 3:
        return Network3(), 'Models\colorization_model3_OG.pth' 
    elif choice == 6:
        return Network6(), 'Models\colorization_model6_OG.pth' # original network no step training, EHHHHH not that great
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
    

def color_frames(image_path, framecount, outputFolder):
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

    # Convert back to BGR for OpenCV saving so we can save the images to our folder, s
    colorized_bgr = cv2.cvtColor((colorized_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # save the colored frame to the folder
    colored_frame_fileName = os.path.join(outputFolder, f"frame_{framecount:05d}.jpg")
    cv2.imwrite(colored_frame_fileName, colorized_bgr)



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
    # City
    # Driving
    video_path = 'TestVideos/catVid_1_1339_1349.mp4'

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
            color_frames(image_path, framecount, coloredFrameFolder) # we know color each frame, save it in the colored folder 
            framecount +=1 # increase frame count to 

    # now we want to reassemble the colored frames into a video and as well delete the frames that we had made
    reassemble_color_video(videoFolder, coloredFrameFolder, video_path)


    for frame in os.listdir(framesFromVideo):
        frame_path = os.path.join(framesFromVideo, frame)
        os.remove(frame_path)

    for frame in os.listdir(coloredFrameFolder):
        frame_path = os.path.join(coloredFrameFolder, frame)
        os.remove(frame_path)




