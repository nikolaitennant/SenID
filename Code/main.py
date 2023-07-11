import os
import tensorflow as tf
from utils import generate_subfolder_name
import numpy as np
from CNN_Model import load_and_process_images, CNN_build_model, CNN_train_model
from visuals import visualize 
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
tf_gpu = len(tf.config.list_physical_devices('GPU'))>0
print("TF GPU is", "available" if tf_gpu else "NOT AVAILABLE")

def args(**kwargs):
    """
    Returns a dictionary containing various configuration options for a model.

    Returns:
       config (dict): A dictionary containing configuration options.
    """
    config = {
        'CNN': 'RN50', # options: 'custom', 'RN50'
        'train_ratio': 0.8,
        'random_state': 42,
        'early_stopping_patience': 4,
        'epochs': 5,
        'batch_size': 64,
        'validation_split': 0.2,
        'output_units': 2,
        'load_model': False,
        'lime_samples': 2,

        #Dataset Combination
        'Mice Train': False,
        'Human Train': True,
        'Mice Test': False,
        'Human Test': True,

        # ResNet50
        'dense_units': 256,
        'dropout_rate': 0.5,
        'regularization_rate': 0.01,
        'optimizer': 'adam',
        'rn50_loss': 'categorical_crossentropy',
        'rn50_activation': 'relu',
        'freeze_layers': -5,
        'metrics': ['accuracy', 'AUC'],
        'final_activation_rn50': 'softmax',

        #Custom CNN
        'cv': 3,
        'filters': [16, 20, 20],
        'kernel_sizes': [5, 5, 5],
        'strides': [2, 1, 1],
        'paddings': ['same', 'same', 'same'],
        'pool_sizes': [3, 2, None],
        'pool_strides': [2, 2, None],
        'units': [100],
        'dropout_rate': 0.3,
        'learning_rate': 1e-5,
        'custom_activation': 'relu',
        'custom_loss' : 'binary_crossentropy',
        'params': { # Grid search parameters
            'filters': [[16, 20, 20]],
            'kernel_sizes': [[5, 5, 5]],
            'strides': [[2, 1, 1]],
            'paddings': [['same', 'same', 'same']],
            'pool_sizes': [[3, 2, None]],
            'pool_strides': [[2, 2, None]],
            'units': [[100]],
            'dropout_rate': [0.3],
            'learning_rate': [1e-5],
            'batch_size': [64],
            'epochs': [1, 15],
            'custom_activation': ['relu'],
            'custom_loss' : ['binary_crossentropy']
        },

        'datagen_params': { #data augmentation
            'rotation_range': 10,
            'zoom_range': 0.1,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'horizontal_flip': False,
            'vertical_flip': False,
            'fill_mode': 'constant',
            'cval': 0.0
        }
    }
    config.update(kwargs)
    return config


def get_selected_samples(directories, config):
    """
    Returns the number of samples to use for training and testing based on the minimum number of samples
    in the given directories and the train_ratio specified in the config dictionary.

    Args:
        directories: A list of strings representing the directories to search for image files.
        config: A dictionary containing configuration options, including the train_ratio.
        
    Returns:
        tuple:
            train_samples (int): The number of samples to use for training.
            test_samples (int): The number of samples to use for testing.
    """
    # Count the number of image files in each directory and store in a list
    num_samples = [len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) 
                        and f.lower().endswith(('.tiff', '.tif', '.jpeg', '.jpg', '.png'))]) for directory in directories]
    
    # Find the minimum number of samples in the directories
    min_samples = min(num_samples)
    
    # Calculate the number of samples to use for training and testing based on the train_ratio in the config dictionary
    train_samples = int(min_samples * config['train_ratio'])
    test_samples = min_samples - train_samples

    return train_samples, test_samples


def main(config):
    """
    Run the entire process of loading data, training, and evaluating the model.

    Args: 
        config (dict): A dictionary containing the configurations of the CNN model and data preprocessing

    Returns: 
        None
    """

    # Check if the CNN configuration is either 'custom' or 'RN50'
    if config['CNN'] not in ['custom', 'RN50']:
        raise ValueError("The 'CNN' configuration should be either 'custom' or 'RN50'")

    # Get the directories for the input data
    Code_dir = os.path.dirname(os.path.abspath(__file__))
    Data_dir = os.path.join(Code_dir, "..", "Data")

    # Choose the dataset combination
    use_mice_train = config['Mice Train']
    use_human_train = config['Human Train']
    use_mice_test = config['Mice Test']
    use_human_test = config['Human Test']

    subfolder_name = generate_subfolder_name(use_mice_train, use_human_train, use_mice_test, use_human_test)
    
    if not config["load_model"]:
    
        # Get the directories for the input data
        Mice_cycling_processed_dir = os.path.join(Data_dir, "Cycling_cells", "Mice", "Cycling_img_processed")
        Mice_senescent_processed_dir = os.path.join(Data_dir, "Senescent_cells","Mice", "Senescent_img_processed")
        Human_cycling_processed_dir = os.path.join(Data_dir, "Cycling_cells", "Human", "Cycling_img_processed")
        Human_senescent_processed_dir = os.path.join(Data_dir, "Senescent_cells","Human", "Senescent_img_processed")

        # Load the datasets
        mice_train_samples, mice_test_samples = get_selected_samples([Mice_cycling_processed_dir, Mice_senescent_processed_dir], config)
        human_train_samples, human_test_samples = get_selected_samples([Human_cycling_processed_dir, Human_senescent_processed_dir], config)
        
        # Load and process the images
        mice_data = load_and_process_images(Mice_cycling_processed_dir, Mice_senescent_processed_dir, 
                                            config, config[ 'train_ratio'], mice_train_samples, mice_test_samples)
        human_data = load_and_process_images(Human_cycling_processed_dir, Human_senescent_processed_dir, 
                                            config, config[ 'train_ratio'], human_train_samples, human_test_samples)

        # Get the minimum sample size among the selected datasets for training
        if use_mice_train and use_human_train:
            train_samples = min(len(mice_data[0]), len(human_data[0]))
        elif use_mice_train:
            train_samples = len(mice_data[0])
        elif use_human_train:
            train_samples = len(human_data[0])

        # Get the minimum sample size among the selected datasets for testing
        if use_mice_test and use_human_test:
            test_samples = min(len(mice_data[1]), len(human_data[1]))
        elif use_mice_test:
            test_samples = len(mice_data[1])
        elif use_human_test:
            test_samples = len(human_data[1])

        # Extract the samples
        train_images_parts = []
        test_images_parts = []
        train_labels_parts = []
        test_labels_parts = []

        # Combine the data
        if use_mice_train:
            train_images_parts.append(mice_data[0][:train_samples])
            train_labels_parts.append(mice_data[2][:train_samples])
        if use_human_train:
            train_images_parts.append(human_data[0][:train_samples])
            train_labels_parts.append(human_data[2][:train_samples])
        if use_mice_test:
            test_images_parts.append(mice_data[1][:test_samples])
            test_labels_parts.append(mice_data[3][:test_samples])
        if use_human_test:
            test_images_parts.append(human_data[1][:test_samples])
            test_labels_parts.append(human_data[3][:test_samples])

        # Combine the data
        train_labels = np.concatenate(train_labels_parts, axis=0)
        test_labels = np.concatenate(test_labels_parts, axis=0)
        train_inputs = np.concatenate(train_images_parts, axis=0)
        test_inputs = np.concatenate(test_images_parts, axis=0)

        # Build the model
        model = CNN_build_model(config)

        # Train the model
        history, best_model = CNN_train_model(model, train_inputs, train_labels, config)  

    else:
        # Load the model directory
        if config['CNN'] == 'RN50':
            model_type = 'ResNet50'
            model_name = "best_model_rn50.h5"
        else: 
            model_type = 'Custom CNN'
            model_name = "best_model_custom.h5"

        model_dir = os.path.join(Code_dir, "..", "Models", model_type, subfolder_name)

        # Check if the model file exists
        if not os.path.exists(os.path.join(model_dir, model_name)):
            raise ValueError("Model file {} does not exist in directory {}".format(model_name, model_dir))
        
        # Load the model test data
        Mice_cycling_test_only = os.path.join(Data_dir, "test_only", "Cycling_cells", "Mice_Cycling")
        Mice_senescent_test_only = os.path.join(Data_dir, "test_only", "Senescent_cells", "Mice_Senescent")
        Human_cycling_test_only = os.path.join(Data_dir, "test_only", "Cycling_cells", "Human_Cycling")
        Human_senescent_test_only = os.path.join(Data_dir, "test_only", "Senescent_cells", "Human_Senescent")

        # Load and process the images
        mice_data = load_and_process_images(Mice_cycling_test_only, Mice_senescent_test_only, 
                                            config, .01)
        human_data = load_and_process_images(Human_cycling_test_only, Human_senescent_test_only, 
                                            config, .01)

        # Get the minimum sample size among the selected datasets for testing
        if use_mice_test and use_human_test:
            test_samples = min(len(mice_data[1]), len(human_data[1]))
        elif use_mice_test:
            test_samples = len(mice_data[1])
        elif use_human_test:
            test_samples = len(human_data[1])

        # Extract the samples
        test_images_parts = []
        test_labels_parts = []

        # Combine the data
        if use_mice_test:
            test_images_parts.append(mice_data[1][:test_samples])
            test_labels_parts.append(mice_data[3][:test_samples])
        if use_human_test:
            test_images_parts.append(human_data[1][:test_samples])
            test_labels_parts.append(human_data[3][:test_samples])

        # Combine the data
        test_labels = np.concatenate(test_labels_parts, axis=0)
        test_inputs = np.concatenate(test_images_parts, axis=0)

        # Load the model history
        with open(os.path.join(model_dir, "history.pickle"), "rb") as history_file:
            history = pickle.load(history_file)

        # Load the model
        model_path = os.path.join(model_dir, model_name)
        best_model = tf.keras.models.load_model(model_path)

    # Visualize the results
    visualize(config, best_model, history, test_inputs, test_labels, subfolder_name) 


if __name__ == '__main__':
    main(config=args())