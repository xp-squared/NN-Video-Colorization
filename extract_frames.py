import cv2 # OpenCV to process the videos
import os # to handle the directory and file operations

# def Process_Video():

# for this small example below, this our test to make sure we can extract frames from the first video
def TestFrameCapture(path, output_folder):
    video = cv2.VideoCapture(path)

    # error opening video
    if not video.isOpened():
        print("Error could not open video: {path}")
        return
    
    # keeping track of how many frames
    frameCount = 0

    # checks whether frames were extracted
    success = 1

    # read first frame from video
    success, frame = video.read()

    # loop till no more frames
    while success:
        # making filename for each frame
        frame_fileName = os.path.join(output_folder, f"frame_{frameCount:05d}.png")

        # saving frame as png
        cv2.imwrite(frame_fileName, frame)
        print(f"Saved Frame: {frame_fileName}")

        # go onto the next frame!
        success, frame = video.read()
        frameCount += 1

    # release video object after processing
    video.release()



if __name__ == '__main__':
    # taking our dataset folder
    data_folder = './data/UCF-101'

    # where our extracted frames will be
    output_folder = './extracted_frames'

    # test function example
    # r is used so the \ are not recognized as escape characters
    test_video = r'data\UCF-101\ApplyEyeMakeup\v_ApplyEyeMakeup_g01_c01.avi'
    test_output_folder = './extracted_frames'
    TestFrameCapture(test_video, test_output_folder)

    # process_video(data_folder, output_folder, frame_rate = 1) # extract 1 frame per second