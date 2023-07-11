import os
from multiprocessing import Pool
import numpy as np
import cv2
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import keras.callbacks
from sklearn.metrics import accuracy_score
import pickle
import tensorflow as tf


def generate_subfolder_name(use_train1, use_train2, use_test1, use_test2):
    """
    Generates a subfolder name based on the given boolean parameters.

    Args:
        use_train1 (bool): Whether to include the 'train_mice' label in the subfolder name.
        use_train2 (bool): Whether to include the 'train_human' label in the subfolder name.
        use_test1 (bool): Whether to include the 'test_mice' label in the subfolder name.
        use_test2 (bool): Whether to include the 'test_human' label in the subfolder name.

    Returns:
       subfolder_name (string): A string representing the subfolder name based on the given parameters.
    """
    # Initialize an empty string to store the subfolder name
    subfolder_name = ''
    
    # If the use_train1 and use_train2 parameters are both True, add 'train_both' to the subfolder name
    if use_train1 and use_train2:
        subfolder_name += 'train_both'
    # Otherwise, if only use_train1 is True, add 'train_mice' to the subfolder name
    elif use_train1:
        subfolder_name += 'train_mice'
    # Otherwise, if only use_train2 is True, add 'train_human' to the subfolder name
    elif use_train2:
        subfolder_name += 'train_human'

    # Add a dash to separate the train and test labels in the subfolder name
    subfolder_name += '-'

    # If the use_test1 and use_test2 parameters are both True, add 'test_both' to the subfolder name
    if use_test1 and use_test2:
        subfolder_name += 'test_both'
    # Otherwise, if only use_test1 is True, add 'test_mice' to the subfolder name
    elif use_test1:
        subfolder_name += 'test_mice'
    # Otherwise, if only use_test2 is True, add 'test_human' to the subfolder name
    elif use_test2:
        subfolder_name += 'test_human'

    # Return the final subfolder name
    return subfolder_name


class CustomModelSaver(keras.callbacks.Callback):
    """
    Keras callback that saves the model after each epoch during training, only if the current epoch's validation loss is lower than the validation loss of the previously saved model.

    Args:
        base_save_path (str): Base directory path to save the model.
        model_type (str): Name of the model to save.
        validation_data (tuple): Tuple of validation data to use while saving the model.

    Methods:
        on_epoch_end(epoch, logs=None):
            Method that is called at the end of each epoch during training.
    """
    # Constructor method that initializes the necessary parameters to save the model.
    def __init__(self, base_save_path, pickle_path, model_type, config, validation_data=None):
        super().__init__()
        self.save_path = base_save_path # Base directory path to save the model.
        self.pickle_save_path = pickle_path # Base directory path to save Pickle.
        self.model_type = model_type # Name of the model to save.
        self.validation_data = validation_data # Validation data to use while saving the model.
        self.model_was_improved = False
        self.config = config  # Save config as an instance variable

    def on_epoch_end(self, epoch, logs=None):
        """ 
        Saves a model at the end of each epoch during training if val loss > current models loss.
        
        Args:
            epoch (int): The current epoch number.
             logs (dict): Dictionary containing the training and validation loss and accuracy values.
              
        Returns:
            None 
        """
        # Get the current validation loss and the validation loss of the previously saved model
        val_loss = logs.get('val_loss')

        # Load the model from the folder if it exists
        if os.path.exists(self.save_path):
            saved_model = tf.keras.models.load_model(self.save_path, compile=False)
            if self.config['CNN'] == 'RN50':
                saved_model.compile(
                    optimizer=self.config['optimizer'],
                    loss=self.config['rn50_loss'],
                    metrics=self.config['metrics']
                )
            elif self.config['CNN'] == 'custom':
                saved_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                    loss=self.config['custom_loss'],
                    metrics=self.config['metrics'])
            saved_model_val_loss = saved_model.evaluate(self.validation_data[0], self.validation_data[1], verbose=0)[0]
        else:
            # Save the initial model if it doesn't exist
            saved_model_val_loss = float('inf')
            print(f"\nInitial {self.model_type} model saving at {self.save_path}")
            self.model.save(self.save_path)

        # Save the current model only if its validation loss is lower than the loaded model
        if val_loss is not None and val_loss < saved_model_val_loss:
            print(f"\nImproved {self.model_type} model found. Saving at {self.save_path}")
            self.model.save(self.save_path)
            self.model_was_improved = True
            
            # Save the history of the model
            with open(os.path.join(self.pickle_save_path, "history.pickle"), "wb") as history_file:
                pickle.dump(self.model.history.history, history_file)

class ImageCropper:
    """ A utility class for cropping images.
    
    Args:
        None

    Methods:
        find_largest_contour(image):
            Finds the largest contour in a grayscale image using OpenCV.

        crop_image(image: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
            Crops an input image around its largest contour and resizes it to the specified target size.

        crop_images(inputs, target_size=(224, 224)):
            Applies crop_image to a list of images using multiprocessing.
    """

    @staticmethod
    def find_largest_contour(image):
        """
        Finds the largest contour in a grayscale image using OpenCV.

        Args:
            image (numpy.ndarray): A grayscale image as a numpy array.

        Returns:
            numpy.ndarray: The largest contour in the image as a numpy array.
        """
        # Ensure the image is in grayscale format and of type np.uint8
        assert len(image.shape) == 2, f"Image must be in grayscale format. Found shape: {image.shape}"
        assert image.dtype == np.uint8, f"Image data type must be np.uint8. Found data type: {image.dtype}"

        # Apply a Gaussian blur to the image to remove noise, and threshold it to create a binary image
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the contours in the binary image and return the largest contour by area
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea)
    

    @staticmethod
    def crop_image(image: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
        """ 
        Crops an input image around its largest contour and resizes it to the specified target size.
        
        Args:
            image (np.ndarray): The input image, which can be grayscale, RGB, or RGBA.
            target_size (tuple[int, int]): A tuple specifying the desired width and height of the resized image, Defaults to (224, 224).
                
        Returns: 
            resized_image (np.ndarray): The cropped and resized image.
        
        """
        # Check if the image has 3 or 4 channels
        if len(image.shape) == 3 and image.shape[2] in [3, 4]:
            # Convert the image to grayscale if it has 3 or 4 channels
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # Use the input image as is if it is already grayscale
            gray_image = image
        else:
            # Raise a ValueError if the input image has an unsupported number of channels
            raise ValueError("Unsupported number of channels in the input image")

        # Find the largest contour in the grayscale image using the `find_largest_contour` method of the `ImageCropper` class
        largest_contour = ImageCropper.find_largest_contour(gray_image)

        # Get the bounding box of the largest contour and crop the image using NumPy array slicing
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]

        # Resize the cropped image to the target size using OpenCV's `cv2.resize` function
        resized_image = cv2.resize(cropped_image, target_size)

        # Return the resized image
        return resized_image


    @staticmethod
    def crop_images(inputs, target_size=(224, 224)):
        """
        Applies crop_image to a list of images using multiprocessing.
        
        Args:
            inputs (list[numpy.ndarray]): A list of images as numpy arrays.
            target_size (tuple[int, int]): A tuple specifying the target size for the cropped and resized images, Default is (224, 224).

        Returns:
            numpy.ndarray: The cropped and resized images as a numpy array.
        """
        
        # Use multiprocessing to apply the `crop_image` method to each image in the input list
        with Pool(os.cpu_count()) as p:
            cropped_images = p.starmap(ImageCropper.crop_image, [(image, target_size) for image in inputs])

        # Stack the cropped images into a numpy array and return it
        return np.stack(cropped_images)


class Explainer:
    """
    This class provides methods for explaining predictions made by a machine learning model using LIME (Local Interpretable Model-Agnostic Explanations).

    Args:
        None

    Methods:
        explain_predictions_lime(model, selected_class_0_samples, selected_class_1_samples, num_samples):
            Uses LIME to explain the predictions made by a machine learning model on two sets of images belonging to two different classes.

        select_random_samples(test_data, test_labels, num_samples_per_class):
            Randomly selects a specified number of samples from each class in the test data.
    """
    def explain_predictions_lime(self, model, selected_class_0_samples, selected_class_1_samples, num_samples):
        """
        Uses LIME to explain the predictions made by a machine learning model on two sets of images belonging to two different classes.

        Args:
            model (object): A machine learning model that has a `predict` method.
            selected_class_0_samples (numpy array): A set of images belonging to class 0.
            selected_class_1_samples (numpy array): A set of images belonging to class 1.
            num_samples (int): The number of samples to use for LIME.

        Returns:
            tuple:
                class_0_explanations (list): A list of LIME explanations for the images belonging to class 0.
                class_1_explanations (list): A list of LIME explanations for the images belonging to class 1.
        """
        # Create a LIME explainer
        explainer = lime_image.LimeImageExplainer()

        # Define the segmentation function
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)

        # Process class 0 samples
        class_0_explanations = []
        for image in selected_class_0_samples:
            explanation = explainer.explain_instance(image, model.predict, top_labels=1, hide_color=None, 
                                                     num_samples=num_samples, segmentation_fn=segmenter)
            class_0_explanations.append(explanation)

        # Process class 1 samples
        class_1_explanations = []
        for image in selected_class_1_samples:
            explanation = explainer.explain_instance(image, model.predict, top_labels=1, hide_color=None, 
                                                     num_samples=num_samples, segmentation_fn=segmenter)
            class_1_explanations.append(explanation)

        return class_0_explanations, class_1_explanations


    def select_random_samples(self, test_data, test_labels, num_samples_per_class):
        """
        Randomly selects a specified number of samples from each class in the test data.

        Args:
            self: (object) The class instance.
            test_data: (numpy array) The test data to select samples from.
            test_labels: (numpy array) The labels for the test data.
            num_samples_per_class: (int) The number of samples to select from each class.

        Returns:
            tuple:
                selected_class_0_samples: (numpy array) The randomly selected samples from class 0.
                selected_class_1_samples: (numpy array) The randomly selected samples from class 1.
        """
        # Find the indices of the two classes (assuming they are labeled as 0 and 1)
        class_0_indices = np.where(test_labels == 0)[0]
        class_1_indices = np.where(test_labels == 1)[0]

        # Select random indices from each class
        selected_class_0_indices = np.random.choice(class_0_indices, num_samples_per_class, replace=False)
        selected_class_1_indices = np.random.choice(class_1_indices, num_samples_per_class, replace=False)

        # Get the samples for the selected indices
        selected_class_0_samples = test_data[selected_class_0_indices]
        selected_class_1_samples = test_data[selected_class_1_indices]

        return selected_class_0_samples, selected_class_1_samples
    

class Prediction:
    """
    This class provides methods for evaluating and making predictions using a trained machine learning model.

    Args:
        None

    Methods:
        baseline_accuracy(y_true):
            Calculates the baseline accuracy given a set of true labels.

        evaluate_model(model, test_inputs, test_labels):
            Evaluates the model on test data.

        make_predictions(model, test_inputs):
            Makes predictions on test data using the trained model.
    """
    def baseline_accuracy(y_true):
        """
        Calculate the baseline accuracy given a set of true labels.

        Args:
            y_true (array-like): The true labels.

        Returns:
           baseline_acc (float): The baseline accuracy.
        """
        # Calculate the most frequent class in the true labels
        most_frequent_class = np.argmax(np.bincount(y_true))

        # Create an array with the same shape as y_true filled with the most frequent class
        baseline_predictions = np.full_like(y_true, most_frequent_class)

        # Calculate the baseline accuracy
        baseline_acc = accuracy_score(y_true, baseline_predictions)

        return baseline_acc
    

    def evaluate_model(model, test_inputs, test_labels):
        """
        Evaluate the model on test data.

        Args:
            model (Model): A trained TensorFlow model.
            test_inputs (numpy.ndarray): Test input data.
            test_labels (numpy.ndarray): Test labels.

        Returns:
            tuple: A tuple containing the following three elements:
                test_loss (float): The value of the test loss for the input data.
                test_acc (float): The value of the test accuracy for the input data.
                test_auc (float): The value of the test AUC for the input data.
        """
        # Evaluate the model on test data
        test_loss, test_acc, test_auc = model.evaluate(test_inputs, test_labels)
        return test_loss, test_acc, test_auc


    def make_predictions(model, test_inputs):
        """
        Make predictions on test data using the trained model.

        Args:
            model (Model): A trained TensorFlow model.
            test_inputs (numpy.ndarray): Test input data.

        Returns:
            tuple: A tuple containing the following two elements:
                y_pred (numpy.ndarray): An array of predicted probabilities for the input data.
                y_pred_binary (numpy.ndarray): An array of binary predictions for the input data.
        """
        # Make predictions using the trained model
        y_pred = model.predict(test_inputs)

        # Convert the predicted probabilities into binary predictions
        y_pred_binary = np.argmax(y_pred, axis=1)
        
        return y_pred, y_pred_binary
