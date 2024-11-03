import os
import shutil # allows more file operations
import random

def split_dataset(input_color_folder,input_grayscale_folder,output_folder,split_ratio = 0.8):
    # create the directories for training and testing data
    train_color_folder = os.path.join(output_folder, 'train/color')
    train_grayscale_folder = os.path.join(output_folder, 'train/grayscale')
    test_color_folder = os.path.join(output_folder, 'test/color')
    test_grayscale_folder = os.path.join(output_folder, 'test/grayscale')

    # Create the train and validation directories if they don't exist
    # exist_ok = true creates folders if they do not exist already 
    os.makedirs(train_color_folder, exist_ok=True)
    os.makedirs(train_grayscale_folder, exist_ok=True)
    os.makedirs(test_color_folder, exist_ok=True)
    os.makedirs(test_grayscale_folder, exist_ok=True)

    # get list of files from color frames folder
    color_frames = os.listdir(input_color_folder)
    print(f"Total files in color folder before filtering: {len(color_frames)}")  # DEBUG

    # Get a list of all .png files from color frames folder and its subdirectories
    color_frames = []
    for root, _, files in os.walk(input_color_folder):
        for file in files:
            if file.endswith('.png'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, input_color_folder)
                color_frames.append((full_path, relative_path))


    print(f"Total color frames after filtering: {len(color_frames)}") # DEBUG
    random.shuffle(color_frames)

    # Calculate the split index based on the given split ratio
    split_index = int(len(color_frames) * split_ratio)
    # Split the files into training and validation sets
    train_files = color_frames[:split_index]  # first 80% of the files
    test_files = color_frames[split_index:]    # reamining files

    # moving training and test files
    for file_path in train_files:
        file_name = os.path.basename(file_path)
        print(f"Moving training file: {file_name}")
        shutil.move(file_path, train_color_folder)
        # Construct the corresponding grayscale file path and move it
        grayscale_file_path = os.path.join(input_grayscale_folder, file_name)
        if os.path.exists(grayscale_file_path):
            shutil.move(grayscale_file_path, train_grayscale_folder)
        else:
            print(f"Warning: Grayscale file for {file_name} not found.")

    for file_path in test_files:
        file_name = os.path.basename(file_path)
        print(f"Moving testing file: {file_name}")
        shutil.move(file_path, test_color_folder)
        # Construct the corresponding grayscale file path and move it
        grayscale_file_path = os.path.join(input_grayscale_folder, file_name)
        if os.path.exists(grayscale_file_path):
            shutil.move(grayscale_file_path, test_grayscale_folder)
        else:
            print(f"Warning: Grayscale file for {file_name} not found.")

    print("Number of training files: " + str(len(train_files)) + "\nNumber of testing files: " + str(len(test_files)))

        

if __name__ == '__main__':
    # get folders for input frames that are grayscale and color
    input_color_folder = './extracted_frames'
    input_grayscale_folder = './grayscale_extracted_frames'

    output_folder = './dataset' # creating the actual training and testing dataset

    print(f"Input color folder exists: {os.path.exists(input_color_folder)}")  # DEBUG
    print(f"Input grayscale folder exists: {os.path.exists(input_grayscale_folder)}")  # DEBUG

    split_dataset(input_color_folder,input_grayscale_folder,output_folder,split_ratio = 0.8)
