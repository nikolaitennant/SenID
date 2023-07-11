import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, average_precision_score, precision_recall_curve, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from utils import Prediction, Explainer

class VisualizationTools:
    """
    This class provides a set of visualization tools for analyzing the performance of classification models. It creates plots
    and visualizations, such as image grids, confusion matrices, training history graphs, ROC curves, precision-recall curves,
    and LIME explanations.

    Args:
        model (str): The model architecture used, such as 'custom', 'RN50', or other specified model names.
        subfolder (str): The name of the subfolder to be created under the model-specific output directory.
        script_dir (str): The directory path of the current script.
        output_dir (str): The directory path for storing output files.

    Methods:
        plotter: Creates a grid of images with predicted and actual labels and returns the plot as a Matplotlib figure.
        visualize_results: Plots a grid of image examples, showing the predicted and actual labels for each example.
        plot_confusion_matrix: Plots a confusion matrix.
        plot_training_history: Plots the training and validation loss and accuracy for each epoch.
        plot_roc_curve: Plots the ROC curve.
        plot_precision_recall_curve: Plots the precision-recall curve.
        plot_lime_explanation: Plots a LIME explanation.
    """
    def __init__(self, model, subfolder):
        # Get the path of the current script and set the output directory to the Visualizations directory one level above
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.script_dir, "..")
        visualizations_dir = os.path.join(self.output_dir, "Visualizations")
        
        # Create the Visualizations directory if it doesn't exist
        os.makedirs(visualizations_dir, exist_ok=True)
        self.output_dir = visualizations_dir

        # Set the output directory based on the model architecture used
        if model == 'custom':
            model_output_dir = os.path.join(self.output_dir, "Custom_CNN")
        elif model == 'RN50':
            model_output_dir = os.path.join(self.output_dir, "ResNet50")
        else:
            raise ValueError("Invalid value for model: {}".format(model))

        # Create the model output directory if it doesn't exist
        os.makedirs(model_output_dir, exist_ok=True)
        self.output_dir = model_output_dir

        # Create a new subfolder based on the input parameter
        subfolder_output_dir = os.path.join(self.output_dir, subfolder)
        os.makedirs(subfolder_output_dir, exist_ok=True)
        self.output_dir = subfolder_output_dir


    def plotter(self, image_indices, label, image_inputs, predicted_labels, image_labels, first_label, second_label):   
        """
        Creates a grid of images with predicted and actual labels and returns the plot as a Matplotlib figure.

        Args:
            image_indices (list of int): Indices of the images to plot.
            label (str): Title of the plot.
            image_inputs (np.ndarray): Array of input images.
            predicted_labels (np.ndarray): Array of predicted labels.
            image_labels (np.ndarray): Array of actual labels.
            first_label (str): Name of the first label.
            second_label (str): Name of the second label.

        Returns:
            fig (matplotlib.figure.Figure): The plot as a Matplotlib figure object.
        """
        # Determine the number of columns (nc) and rows (nr) for the grid
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        # Create a new Matplotlib figure
        fig = plt.figure()
        # Set the title of the figure
        fig.suptitle(f"{label} Examples\nPL = Predicted Label\nAL = Actual Label")

        # Loop over the image indices and add each image to the plot
        for i in range(len(image_indices)):
            ind = image_indices[i]
            # Add a new subplot to the figure and display the image
            ax = fig.add_subplot(nr, nc, i + 1)
            ax.imshow(image_inputs[ind], cmap='gray')


            # Set the predicted label (pl) and actual label (al)
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label

            # Hide the x and y tick labels and tick marks
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

        # Return the figure object
        return fig


    def visualize_results(self, image_inputs, probabilities, image_labels, first_label, second_label):
        """
        Plots a grid of image examples, showing the predicted and actual labels for each example.

        Args:
            image_inputs (numpy.ndarray): A 4D numpy array of shape (num_examples, height, width, channels)
                containing the images to display.
            probabilities (numpy.ndarray): A 2D numpy array of shape (num_examples, num_classes) containing
                the predicted class probabilities for each example.
            image_labels (numpy.ndarray): A 2D numpy array of shape (num_examples, num_classes) containing
                the true class labels for each example.
            first_label (str): The name of the first class.
            second_label (str): The name of the second class.

        Returns:
            None
        """
        # Get the predicted labels from the predicted probabilities
        predicted_labels = np.argmax(probabilities, axis=1)
        # Get the number of images in the input array
        num_images = image_inputs.shape[0]

        # Create two lists: one for correctly classified examples, and another for incorrectly classified examples
        correct = []
        incorrect = []

        for i in range(num_images):
            # Determine if the example was correctly or incorrectly classified
            if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
                correct.append(i)
            else:
                incorrect.append(i)

        # Use the plotter method to create two separate Matplotlib figures.
        # one for correctly classified examples, and another for incorrectly classified examples
        fig_correct = self.plotter(correct, 'Correct', image_inputs, predicted_labels,
                                   image_labels, first_label, second_label)
        fig_incorrect = self.plotter(incorrect, 'Incorrect', image_inputs, predicted_labels, 
                                    image_labels, first_label, second_label)

        # Save the two figures to separate files in the output directory
        correct_output_path = os.path.join(self.output_dir, 'correct_examples.png')
        fig_correct.savefig(correct_output_path)

        incorrect_output_path = os.path.join(self.output_dir, 'incorrect_examples.png')
        fig_incorrect.savefig(incorrect_output_path)

        # Close all figures to free up memory
        plt.close('all')


    def create_confusion_matrix(self, y_true, y_pred, class_names, file_name):
        """
        Creates and saves a confusion matrix plot based on the true and predicted labels.

        Args:
            y_true (array-like): Ground truth (correct) target values.
            y_pred (array-like): Estimated targets as returned by a classifier.
            class_names (list): List of class names (strings) in order of the confusion matrix.
            file_name (str): File name to save the confusion matrix plot.

        Returns:
            None
        """
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
       
        # Create a ConfusionMatrixDisplay object using the computed confusion matrix and class names
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        
        # Create a new Matplotlib figure with a given size
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use the ConfusionMatrixDisplay object to plot the confusion matrix
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
        
        # Set the x- and y-axis labels and title of the plot
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        # Save the plot to a file in the output directory
        output_file_path = os.path.join(self.output_dir, file_name)
        fig.savefig(output_file_path)
        
        # Close the figure to free up memory
        plt.close(fig)


    def plot_history(self, history, file_name):
        """ 
        Plots the training and validation loss, accuracy, AUC (Area Under the ROC Curve), for each epoch in a given Keras training history.

        Args:
            history (tensorflow.python.keras.callbacks.History or dict): A Keras training history object or a dictionary containing training history data.
            file_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        if isinstance(history, dict):
            history_data = history
        else:
            history_data = history.history

        # Define the metrics to plot and their titles and y-axis labels
        metrics = [
            ('loss', 'Model Loss', 'Loss'),
            ('accuracy', 'Model Accuracy', 'Accuracy'),
            ('auc', 'Model AUC', 'AUC')
        ]

        # Determine the number of rows and columns for the subplot grid
        num_metrics = len(metrics)
        ncols = 3
        nrows = (num_metrics + ncols - 1) // ncols

        # Create a new Matplotlib figure with the appropriate number of subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
        axes = axes.flatten()

        # For each metric, plot the train and validation data, and set the title and y-axis label
        for i, (metric, title, ylabel) in enumerate(metrics):
            ax = axes[i]
            ax.plot(range(1, len(history_data[metric]) + 1), history_data[metric])
            ax.plot(range(1, len(history_data[f'val_{metric}']) + 1), history_data[f'val_{metric}'])
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Epoch')
            ax.legend(['Train', 'Validation'], loc='best')

        # Remove any extra subplots
        for i in range(num_metrics, len(axes)):
            fig.delaxes(axes[i])

        # Save the figure to a file in the output directory and close the figure to free up memory
        output_file_path = os.path.join(self.output_dir, file_name)
        plt.savefig(output_file_path)
        plt.close()


    def plot_roc_curve(self, y_true, y_pred_prob, file_name):
        """
        Plots the receiver operating characteristic (ROC) curve for a binary classification problem.

        Args:
            y_true (array-like): Ground truth (correct) target values.
            y_pred_prob (array-like): Probability estimates of the positive class as returned by a classifier.
            file_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        # Compute the false positive rate, true positive rate, and thresholds for the ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        
        # Compute the area under the ROC cu rve
        roc_auc = auc(fpr, tpr)

        # Create a new Matplotlib figure and plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # Save the plot to a file in the output directory and close the figure to free up memory
        output_file_path = os.path.join(self.output_dir, file_name)
        plt.savefig(output_file_path)
        plt.close()


    def plot_precision_recall_curve(self, y_true, y_pred_prob, file_name):
        """
        Plots the precision-recall curve for a binary classification problem.

        Args:
            y_true (array-like): Ground truth (correct) target values.
            y_pred_prob (array-like): Probability estimates of the positive class as returned by a classifier.
            file_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        # Compute the precision, recall, and thresholds for the precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        average_precision = average_precision_score(y_true, y_pred_prob)

        # Create a new Matplotlib figure and plot the precision-recall curve
        plt.figure()
        plt.step(recall, precision, where='post', label='Precision-Recall curve')

        # Compute the baseline precision-recall curve
        no_skill = len(y_true[y_true==1]) / len(y_true)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

        # Add labels and legend to the plot
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        plt.legend()
  
        # Save the plot to a file in the output directory and close the figure to free up memory
        output_file_path = os.path.join(self.output_dir, file_name)
        plt.savefig(output_file_path)
        plt.close()


    def visualize_lime_explanations(self, class_0_explanations, class_1_explanations, class_0_name='Senescent', class_1_name='Healthy'):
        """
        Visualize LIME explanations for a binary classification model.

        Args:
            class_0_explanations (list): List of LIME explanations for class 0.
            class_1_explanations (list): List of LIME explanations for class 1.
            class_0_name (str): Name of class 0 (default: 'Senescent').
            class_1_name (str): Name of class 1 (default: 'Healthy').

        Returns:
            None
        """
        
        # Iterate over the two classes
        for idx, explanations in enumerate([(class_0_explanations, class_0_name), (class_1_explanations, class_1_name)]):
            
            # Determine the number of images to plot and set the figure size accordingly
            num_images = len(explanations[0])
            fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
            
            # Plot the LIME explanations for the current class
            for img_idx, explanation in enumerate(explanations[0]):
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, 
                                                            num_features=5, hide_rest=False)
                axes[img_idx].imshow(temp)
                axes[img_idx].set_title(f"{explanations[1]} Cell {img_idx + 1}")
            
            # Adjust the layout and save the figure to disk
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"lime_explanation_{explanations[1]}.png"))
            plt.close()


def visualize(config, best_model, history, test_inputs, test_labels, subfolder_name):
    """
    This function visualizes the training history, evaluates the model on test data, and generates various performance metrics and plots.

    Args:
        config: Configuration dictionary containing the model type and other parameters.
        best_model: The best trained model for evaluation.
        history: The training history containing the loss and accuracy data.
        test_inputs: The inputs for the test dataset.
        test_labels: The labels for the test dataset.
        subfolder_name: The name of the subfolder where the output files will be saved.
    Returns: 
        None

    Visualizations generated by this function include:

    Model loss history
    Confusion matrix
    Results visualization for cycling and senescene
    ROC curve
    Precision-recall curve
    LIME explanations for model predictions
    Performance metrics generated by this function include:

    Test accuracy
    Test precision
    Test recall
    Test F1 score
    Test AUC
    Baseline accuracy
    """
    # Visualize the training history
    visual_tools = VisualizationTools(model=config['CNN'], subfolder=subfolder_name)
    visual_tools.plot_history(history, "model_loss.png")# Load and process images

    # Evaluate the model on test data
    test_loss, test_acc, test_auc = Prediction.evaluate_model(best_model, test_inputs, test_labels)
    print(f"Eval accuracy: {test_acc}")
    print(f"Eval loss: {test_loss}")
    print(f"Eval AUC: {test_auc}")

    # Make predictions and convert them to binary format
    y_pred, y_pred_binary = Prediction.make_predictions(best_model, test_inputs)
    y_true_binary = np.argmax(test_labels, axis=1)
    test_inputs = (test_inputs - test_inputs.min()) / (test_inputs.max() - test_inputs.min())
    positive_class_prob = y_pred[:, 1]

    #LIME
    lime = Explainer()
    num_samples_per_class = 15
    selected_class_0_samples, selected_class_1_samples = lime.select_random_samples(test_inputs, test_labels, num_samples_per_class)
    class_0_explanations, class_1_explanations = lime.explain_predictions_lime(best_model, selected_class_0_samples, 
                                                                                selected_class_1_samples, num_samples=config['lime_samples'])
    class_0_explanations = np.array(class_0_explanations)
    class_1_explanations = np.array(class_1_explanations)

    # Visualize the results
    class_labels = ["Cycling", "Senescene"]
    visual_tools.create_confusion_matrix(y_true_binary, y_pred_binary, class_labels, "confusion_matrix.png")
    visual_tools.visualize_results(test_inputs.squeeze(), y_pred, test_labels, "Cycling", "Senescene")
    visual_tools.plot_roc_curve(y_true_binary, positive_class_prob, "roc_curve.png")
    visual_tools.plot_precision_recall_curve(y_true_binary, positive_class_prob, "precision_recall_curve.png")
    visual_tools.visualize_lime_explanations(class_0_explanations.squeeze(), class_1_explanations.squeeze(), 'Senescent', 'Cycling')
    
    # Print the classification report and various performance metrics
    print(classification_report(y_true_binary, y_pred_binary))
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, average='macro')
    recall = recall_score(y_true_binary, y_pred_binary, average='macro')
    f1 = f1_score(y_true_binary, y_pred_binary, average='macro')
    auc = roc_auc_score(y_true_binary, positive_class_prob, average='macro')
    baseline_acc = Prediction.baseline_accuracy(y_true_binary)
    print(f"Test Accuracy: {accuracy:.4%}, Test Precision: {precision:.4%}, Test Recall: {recall:.4%}, Test F1: {f1:.4%}, Test AUC: {auc:.4%}, Baseline Accuracy: {baseline_acc:.4%}")
