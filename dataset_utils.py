import os  # Import os to work with file paths and directories

def count_videos(data_folder):
    
    video_count = 0  # Variable to store the count of video files

    # Use os.walk to recursively traverse through the dataset folder
    print("Here are a few sample videos from the dataset: \n")
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            # Check if the current file is a video file (with .avi extension)
            if file.endswith(".avi"):
                video_count += 1  # Increment the count if a video file is found
                # Show a few file names just to see we are moving through it and they are accessible
                if (video_count % 1000 == 0):
                    print(file)
    
    return video_count  # Return the total count of video files

# Testing to see if we get correct amount of video files
if __name__ == "__main__":
    data_folder = './data/UCF-101'

    total_videos = count_videos(data_folder)
    
    print(f"\n\nTotal number of videos in the dataset: {total_videos}")
    if (total_videos == 13320):
        print("Dataset has been properly extracted.\n")
    
    else:
        print("Error may have occured when dataset was extracted.\n")
    
    

