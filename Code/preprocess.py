#################
#
# This script splits images (40x zoom) into squares of 200x200 pixels surronding a cell.
# Directories are cycled through so that all images within a directory are split
#
#################

from cellpose import models
from skimage.io import imread, imsave
import numpy as np
import os
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import random
import hashlib
from PIL import Image



def normalize_and_convert_to_uint8(image):
    """
    Normalize the input image and convert it to uint8 format.

    Args:
        image (numpy.ndarray): Input image in float64 format.

    Returns:
        numpy.ndarray: Normalized image in uint8 format. The pixel values are scaled to the range [0, 255].
    """
    normalized_image = np.clip(image, 0, None)  # Make sure all values are positive
    normalized_image = (normalized_image / np.max(normalized_image)) * 255  # Normalize pixel values to the range 0-255
    return normalized_image.astype(np.uint8)


def split_images(group1_directory, group2_directory, out1_dir, out2_dir):
    """
    Split images into smaller pieces using the provided directories.

    Args:
        group1_directory: The path to the directory containing the Group 1 images.
        group2_directory: The path to the directory containing the Group 2 images.
        out1_dir: The output directory for the Group 1 split images.
        out2_dir: The output directory for the Group 2 split images.

    Returns: 
        None
    """
    # Get the file names for each group
    group1_files = [f for f in os.listdir(group1_directory) if os.path.isfile(os.path.join(group1_directory, f))]
    group2_files = [f for f in os.listdir(group2_directory) if os.path.isfile(os.path.join(group2_directory, f))]
    all_files = group1_files + group2_files

    # Initialize the Cellpose model for nuclei segmentation
    model = models.Cellpose(gpu=True, model_type='nuclei')

    # Loop through all the files
    for file in all_files:
        # Determine the file path based on the group it belongs to
        file_path = os.path.join(group1_directory, file) if file in group1_files else os.path.join(group2_directory, file)
        print(f"Processing file: {file}")
        print(f"File path: {file_path}")
       
        # Read the multichannel image and extract the first channel
        multichannel_image = imread(file_path)

        if multichannel_image.shape[-1] == 3:
            multichannel_image = np.transpose(multichannel_image, (2, 0, 1))

        single_channel_image = multichannel_image[0, :, :]  # Change the index if you want to use a different channel

        # Process the image using the Cellpose model
        channels = [0, 0]  # This means we are processing single channel greyscale images.
        label_image, _, _, _ = model.eval(single_channel_image, diameter=None, channels=channels)
        print(f"Processed image using Cellpose model for file: {file}")

        # Get the number of nuclei in the image
        num_nuclei = label_image.max()
        print(f"Number of nuclei found in file {file}: {num_nuclei}")

        # Loop through each nucleus
        for nucleus_index in range(1, num_nuclei + 1):
            # Get the row and column indices of the nucleus pixels
            rows, cols = np.where(label_image == nucleus_index)

            # Initialize the cropped image with zeros
            cropped_image = np.zeros(shape=(200, 200))

            # Calculate the row and column boundaries, differences, and shifts
            row_max, row_min = max(rows), min(rows)
            col_max, col_min = max(cols), min(cols)
            row_diff, col_diff = row_max - row_min, col_max - col_min
            row_shift, col_shift = math.ceil((200 - row_diff) / 2), math.ceil((200 - col_diff) / 2)

            # Copy the nucleus pixels from the original image to the cropped image
            for row, col in zip(rows, cols):
                cropped_row = row - row_min + row_shift
                cropped_col = col - col_min + col_shift
                cropped_image[cropped_row, cropped_col] = single_channel_image[row, col]

            # Save the cropped nucleus image
            output_directory = out1_dir if file in group1_files else out2_dir
            output_filename = f"{file[:-4]}_nucleus_{nucleus_index}.jpeg"
            imsave(os.path.join(output_directory, output_filename), normalize_and_convert_to_uint8(cropped_image))
            print(f"Saved cropped nucleus image for file {file}: {output_filename}")


def collect_images(group1_dir, group2_dir, train_ratio=0.8, random_state=42, train_samples_species=None, test_samples_species=None):
    """
    Collects images from the provided directories, splits them into training, validation, and test sets, and generates corresponding labels.

    Args:
        group1_dir (str): Directory containing Group 1 images.
        group2_dir (str): Directory containing Group 2 images.
        train_ratio (float, optional): Proportion of the dataset to include in the train set. Defaults to 0.8.
        random_state (int, optional): Seed value used for random number generation. Defaults to 42.
        train_samples_species (int, optional): Number of train samples for each species. If specified, limits the number of samples for both groups.
        test_samples_species (int, optional): Number of test samples for each species. If specified, limits the number of samples for both groups.

    Returns:
        tuple:
            train_inputs (numpy.ndarray): An array of training inputs for CNN usage.
            test_inputs (numpy.ndarray): An array of test inputs for CNN usage.
            train_labels (numpy.ndarray): An array of training labels for CNN usage.
            test_labels (numpy.ndarray): An array of test labels for CNN usage.
    """
    # List filenames for both groups
    onlyfiles_group1 = [f for f in os.listdir(group1_dir) if os.path.isfile(os.path.join(group1_dir, f)) and f.lower().endswith(('.tiff', '.tif', '.jpeg', '.jpg', '.png'))]
    onlyfiles_group2 = [f for f in os.listdir(group2_dir) if os.path.isfile(os.path.join(group2_dir, f)) and f.lower().endswith(('.tiff', '.tif', '.jpeg', '.jpg', '.png'))]

    # If train_samples_species and test_samples_species are specified, limit the number of samples for both groups
    if train_samples_species is not None and test_samples_species is not None:
        random.seed(random_state) # Set seed for reproducibility
        onlyfiles_group1 = random.sample(onlyfiles_group1, train_samples_species + test_samples_species)
        onlyfiles_group2 = random.sample(onlyfiles_group2, train_samples_species + test_samples_species)

    # Create empty arrays for images and labels
    result_arr_group1, labels_group1 = [], []
    result_arr_group2, labels_group2 = [], []

    # Read Group 1 images and labels
    for file in onlyfiles_group1:
        image = imread(os.path.join(group1_dir, file))
        result_arr_group1.append(image)
        labels_group1.append(0)

    # Read Group 2 images and labels
    for file in onlyfiles_group2:
        image = imread(os.path.join(group2_dir, file))
        result_arr_group2.append(image)
        labels_group2.append(1)

    # Combine images and labels for both groups
    images = result_arr_group1 + result_arr_group2
    labels = labels_group1 + labels_group2

    # Split the data into train and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=train_ratio, random_state=random_state, stratify=labels)

    # Convert lists to numpy arrays
    train_images = np.stack(train_images, axis=0)
    test_images = np.stack(test_images, axis=0)
    train_labels = np.array(train_labels).reshape(-1, 1)
    test_labels = np.array(test_labels).reshape(-1, 1)

    return train_images, test_images, train_labels, test_labels


def get_directories(species):
    """
    Returns the paths for directories containing input and output files for image processing.

    Args:
        species (str): The species to process images for. Should be either "human" or "mice".

    Returns:
        tuple:
            group1_directory (str): The path to group 1 input directory.
            group2_directory (str): The path to group 2 input directory.
            out1_dir (str): The path to group 1 output directory.
            out2_dir (str): The path to group 2 output directory.
            output_dir (str): The path to the output directory for the feature set CSV file.
    """
    # Check if the species is valid
    if species.lower() not in ['human', 'mice']:
        raise ValueError("Invalid species. Please choose either 'human' or 'mice'.")

    # Get the path to the current directory and the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(script_dir, "..", "Data")
    
    # Set the input and output directories for both groups
    out1_dir = os.path.join(data, "Cycling_cells", species, "Cycling_img_processed")
    out2_dir = os.path.join(data, "Senescent_cells", species, "Senescent_img_processed")
    group1_directory = os.path.join(data, "Cycling_cells", species, "Cycling_img_raw")
    group2_directory = os.path.join(data, "Senescent_cells", species, "Senescent_img_raw")
    
    # Set the output directory for the feature set CSV file
    output_dir = os.path.join(script_dir, f"features_{species.lower()}.csv")

    return group1_directory, group2_directory, out1_dir, out2_dir, output_dir


def find_duplicates(directory):
    """ 
    Uses a hash table to find duplicate images in a directory
    Args:
        directory (str): The directory to search for duplicates in
    Returns:
        None
    """
    hash_dict = {}
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpeg')]

    for image_file in image_files:
        with Image.open(os.path.join(directory, image_file)) as img:
            temp_hash = hashlib.md5(img.tobytes())
            if temp_hash in hash_dict:
                print(f"Duplicate found: {image_file} and {hash_dict[temp_hash]}")
            else:
                hash_dict[temp_hash] = image_file


#species = # Human or Mice
#group1_directory, group2_directory, out1_dir, out2_dir, output_dir = get_directories(species)
#split_images(group1_directory, group2_directory, out1_dir, out2_dir)
