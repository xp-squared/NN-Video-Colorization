import cv2 # OpenCV to process the videos
import os # to handle the directory and file operations

def FrameCapture(video_path, output_folder): 
    video = cv2.VideoCapture(video_path)

    # error opening video
    if not video.isOpened():
        print(f"Error could not open video: {video_path}")
        return
    
    # create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # keeping track of how many frames
    frameCount = 0

    # read first frame from video
    # will check if success was success by the loop continuing
    success, frame = video.read()

    # loop till no more frames
    while success:
        # making filename for each frame

        # too many frames previously not enough change over the video, instead of 1 frame each time lets use each 10th frame of video
        if (frameCount % 10 == 0): 
            frame_fileName = os.path.join(output_folder, f"frame_{frameCount:05d}.png")

            # saving frame as png
            cv2.imwrite(frame_fileName, frame)
            print(f"Saved Frame: {frame_fileName}")

            # go onto the next frame!
            success, frame = video.read()
        frameCount += 1

    # release video object after processing
    video.release()


def Process_Video(data_folder, output_folder):
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
                relative_path = os.path.relpath(root,data_folder)
                video_output_folder = os.path.join(output_folder, relative_path, os.path.splitext(file)[0])

                # to see if videos are being processed
                print(f"Processing video: {video_path}")

                # Extract frames from the video
                FrameCapture(video_path, video_output_folder)

    

if __name__ == '__main__':
    # taking our dataset folder
    data_folder = './data/UCF-101'

    # where our extracted frames will be
    output_folder = './extracted_frames'


    
    # test function example !!!
    # r is used so the \ are not recognized as escape characters
    # test_video = r'data\UCF-101\ApplyEyeMakeup\v_ApplyEyeMakeup_g01_c01.avi'
    # test_output_folder = './extracted_frames'
    # TestFrameCapture(test_video, test_output_folder)
    

    Process_Video(data_folder, output_folder)