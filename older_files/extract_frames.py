import cv2 # OpenCV to process the videos
import os # to handle the directory and file operations

def FrameCapture(video_path, color_output_folder, grayscale_output_folder, resize_dim=(224, 224)): 
    video = cv2.VideoCapture(video_path)

    # error opening video
    if not video.isOpened():
        print(f"Error could not open video: {video_path}")
        return
    
    # create output folder if it does not exist
    # Create output folders if they do not exist
    if not os.path.exists(color_output_folder):
        os.makedirs(color_output_folder)
    if not os.path.exists(grayscale_output_folder):
        os.makedirs(grayscale_output_folder)

    
    # keeping track of how many frames
    frameCount = 0

    # read first frame from video
    # will check if success was success by the loop continuing
    success, frame = video.read()

    # loop till no more frames
    while success:
        # too many frames previously not enough change over the video, instead of 1 frame each time lets use each 10th frame of video
        if (frameCount % 10 == 0): 
            # Resize frame if resize_dim is specified
            # CNN models require fixed-size inputs, and having images with varying dimensions can lead to
            #  complications during training. By resizing all images to a square shape, we ensure that they
            #  have the same width and height, which simplifies the data handling and processing steps.
            # https://eitca.org/artificial-intelligence/eitc-ai-dltf-deep-learning-with-tensorflow/using-convolutional-neural-network-to-identify-dogs-vs-cats/introduction-and-preprocessing/examination-review-introduction-and-preprocessing/why-is-it-necessary-to-resize-the-images-to-a-square-shape/#:~:text=CNN%20models%20require%20fixed%2Dsize,data%20handling%20and%20processing%20steps.
            if resize_dim:
                frame = cv2.resize(frame, resize_dim)

            color_frame_fileName = os.path.join(color_output_folder, f"frame_{frameCount:05d}.png")
            # save frame as png
            cv2.imwrite(color_frame_fileName, frame)
            print(f"Saved Color Frame: {color_frame_fileName}")

            # convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # save the grayscale frame
            grayscale_frame_fileName = os.path.join(grayscale_output_folder, f"frame_{frameCount:05d}.png")
            cv2.imwrite(grayscale_frame_fileName, gray_frame)

        # go onto the next frame!
        success, frame = video.read()
        frameCount += 1

    # release video object after processing
    video.release()


def Process_Video(data_folder, color_output_folder, grayscale_output_folder):
    # we want to loop through every video folder in the data

    # for the sake of testing we will only test for a few videos
    # defining a counter for now this will be removed

    for root, dirs, files in os.walk(data_folder):
    # root is the current folder, we want to go into that folder
    # print(root) will output something like: ./data/UCF-101\ApplyEyeMakeup
    # print(files) will show [...] filled with each video for 1 topic,
        for file in files:
            if file.endswith(".avi"):
                # get full video path for current file
                video_path = os.path.join(root,file)

                # now we want to create a corresponding output folder for that specific video
                relative_path = os.path.relpath(root, data_folder)
                video_color_output_folder = os.path.join(color_output_folder, relative_path, os.path.splitext(file)[0])
                video_grayscale_output_folder = os.path.join(grayscale_output_folder, relative_path, os.path.splitext(file)[0])

                # to see if videos are being processed
                print(f"Processing video: {video_path}")

                # Extract frames from the video in both color and grayscale
                FrameCapture(video_path, video_color_output_folder, video_grayscale_output_folder)

    

if __name__ == '__main__':
    # taking our dataset folder
    data_folder = './data/UCF-101'

    # where our extracted frames will be
    color_output_folder = './extracted_frames'
    grayscale_output_folder = './grayscale_extracted_frames'
    
    # test function example !!!
    # r is used so the \ are not recognized as escape characters
    # test_video = r'data\UCF-101\ApplyEyeMakeup\v_ApplyEyeMakeup_g01_c01.avi'
    # test_output_folder = './extracted_frames'
    # TestFrameCapture(test_video, test_output_folder)
    

    Process_Video(data_folder, color_output_folder, grayscale_output_folder)