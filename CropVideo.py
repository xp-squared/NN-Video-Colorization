import cv2 # OpenCV to process the videos

def extract_video(input_video, output_video, start_time, end_time):
    # open input video for reading
    video = cv2.VideoCapture(input_video)

    # see if video was opened successfully
    if not video.isOpened():
        print(f"Error could not open video: {input_video}")
        return

    
    fps = video.get(cv2.CAP_PROP_FPS) # getting the frames per second
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT) # total frames of a video
    duration = frame_count / fps # total duration of the vid in seconds
    # width and height of each video frame
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # Calculate frame numbers
    start_frame = int(start_time * fps) # starting frame index
    end_frame = int(end_time * fps) # ending frame index

    # Set the starting frame, this is the frame that we wil start to read from
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height)) # output video!

    current_frame = start_frame

    # while current frame is less then the end frame
    while video.isOpened() and current_frame <= end_frame:
        ret, frame = video.read()
        if not ret: # if frame retrieval failed we will break
            break
        # write the frame to the output video
        out.write(frame)
        current_frame += 1

    # Release resources and close them
    video.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video segment saved to {output_video}")


if __name__ == '__main__':
    # TestVideos/catvid_og_20Min.mp4
    input_video = 'TestVideos\SteamboatWill.mp4' ### 
    output_video = 'TestVideos\Mickey.mp4' ### 
    # made the times I want to crop into seconds
    start_time = 5 * 60 + 0   # 13 minutes * 60 + 39 seconds = 819 seconds, 13 minutes 39 seconds
    end_time = 5 * 60 + 5     # 13 minutes * 60 + 49 seconds = 829 seconds, 13 minutes 49 seconds     

    extract_video(input_video, output_video, start_time, end_time)