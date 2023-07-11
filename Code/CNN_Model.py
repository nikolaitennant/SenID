import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from preprocess import collect_images
from utils import ImageCropper, generate_subfolder_name
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import CustomModelSaver


def load_and_process_images(out1_dir, out2_dir, config, train_ratio = None, train_samples_species = None, test_samples_species = None):
    """
    Load and process images, splitting them into train and test sets.

    Args:
        out1_dir (str): Directory path for cycling cell images.
        out2_dir (str): Directory path for senescent cell images.
        config (dict): Dictionary containing configuration parameters for the preprocessing and model training steps.
        train_ratio (float, optional): Ratio for the train-test split. Defaults to 0.8.
        train_samples_species (list, optional): List of species to use for training, if only a subset is desired. Defaults to empty list, which indicates use of all available species.
        test_samples_species (list, optional): List of species to use for testing, if only a subset is desired. Defaults to empty list, which indicates use of all available species.

    Returns:
        tuple:
            train_inputs (numpy.ndarray): An array of training inputs for CNN usage.
            test_inputs (numpy.ndarray): An array of test inputs for CNN usage.
            train_labels (numpy.ndarray): An array of training labels for CNN usage.
            test_labels (numpy.ndarray): An array of test labels for CNN usage.
    """
    # Collect images and split them into train and test sets
    train_inputs, test_inputs, train_labels, test_labels = collect_images(out1_dir, 
                                                                          out2_dir, 
                                                                          train_ratio=train_ratio, 
                                                                          random_state=config["random_state"], 
                                                                          train_samples_species=train_samples_species, 
                                                                          test_samples_species=test_samples_species)

    # Convert labels to categorical format
    train_labels = to_categorical(train_labels, num_classes=2)
    test_labels = to_categorical(test_labels, num_classes=2)

    # Crop images and preprocess inputs
    train_inputs = ImageCropper.crop_images(train_inputs)
    test_inputs = ImageCropper.crop_images(test_inputs)

    # Preprocess inputs based on whether a custom CNN is being used
    if config['CNN'] == 'custom':
        
        # Reshape inputs and normalize pixel values
        train_inputs = np.repeat(np.expand_dims(train_inputs.astype(np.float32) / 255.0, axis=-1), 3, axis=-1)
        test_inputs = np.repeat(np.expand_dims(test_inputs.astype(np.float32) / 255.0, axis=-1), 3, axis=-1) 
    else:
        # Repeat grayscale images three times to convert them to RGB format
        train_inputs = np.repeat(np.expand_dims(train_inputs, axis =-1), 3, axis=-1)
        test_inputs = np.repeat(np.expand_dims(test_inputs, axis =-1), 3, axis=-1)
        
        # Preprocess inputs using the VGG16 preprocessing function
        train_inputs = preprocess_input(train_inputs)
        test_inputs = preprocess_input(test_inputs)

    return train_inputs, test_inputs, train_labels, test_labels


def create_custom_model(config, filters, kernel_sizes, strides, paddings, pool_sizes, pool_strides, units, dropout_rate, learning_rate, batch_size, epochs, custom_activation, custom_loss, input_shape=(224, 224, 3)):
    """
    Creates a custom convolutional neural network model using the specified hyperparameters.

    Args: 
        config (dict): A dictionary containing the hyperparameters for the model.
        input_shape (tuple): A tuple specifying the shape of the input images.
    Returns:
        model (tf.keras.Model): A compiled convolutional neural network model.
    """
    # create model
    model = tf.keras.Sequential()

    # convolutional blocks
    for i in range(3):
        model.add(tf.keras.layers.Conv2D(filters=config['filters'][i], kernel_size=config['kernel_sizes'][i], strides=config['strides'][i], padding=config['paddings'][i], input_shape=input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        if config['pool_sizes'][i] is not None:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=config['pool_sizes'][i], strides=config['pool_strides'][i], padding='same'))

    # Fully connected layers
    model.add(tf.keras.layers.Flatten())
    for unit in config['units']:
        model.add(tf.keras.layers.Dense(units=unit, activation=config['custom_activation']))
        model.add(tf.keras.layers.Dropout(rate=config['dropout_rate']))
    model.add(tf.keras.layers.Dense(units=config['output_units'], activation='softmax'))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=config['custom_loss'],
        metrics=config['metrics']
    )

    return model

def CNN_build_model(config):
    """
    Build a CNN model based on the ResNet50 architecture.

    Args:
        config (dict): A dictionary containing the configuration options for the model.

    Returns:
        model (tf.keras.Model): A compiled TensorFlow model.
    """
    # Create a custom CNN model if specified
    if config['CNN'] == 'custom':
        model = KerasClassifier(build_fn=create_custom_model, config=config, verbose=0)
    else:
        # Create a base ResNet50 model with ImageNet weights
        base_model = ResNet50(weights='imagenet', include_top=False)

        # Freeze all layers except the last 5
        for layer in base_model.layers[:config['freeze_layers']]:
            layer.trainable = False
        for layer in base_model.layers[config['freeze_layers']:]:
            layer.trainable = True

        # Add custom layers for classification
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(config['dense_units'], activation=config['rn50_activation'])(x)
        x = Dropout(config['dropout_rate'])(x)
        predictions = Dense(config['output_units'], activation=config['final_activation_rn50'], kernel_regularizer=regularizers.l2(config['regularization_rate']))(x)

        # Build and compile the final model
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=config['optimizer'], loss=config['rn50_loss'], metrics=config['metrics'])

    return model


def create_best_model(config):
    """
    Creates a custom convolutional neural network model using the best hyperparameters found by GridSearchCV.

    Args: 
        config (dict): A dictionary containing the best hyperparameters for the model.
    Returns:
        model (tf.keras.Model): A compiled convolutional neural network model.
    """
    # parameters extracted from the configuration
    filters = config['filters']
    kernel_sizes = config['kernel_sizes']
    strides = config['strides']
    paddings = config['paddings']
    pool_sizes = config['pool_sizes']
    pool_strides = config['pool_strides']
    units = config['units']
    dropout_rate = config['dropout_rate']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epochs = config['epochs']
    custom_activation = config['custom_activation']
    custom_loss = config['custom_loss']
    input_shape =  (224, 224, 3)

    # create the model using these parameters
    best_model = create_custom_model(config, filters, kernel_sizes, strides, paddings, pool_sizes, pool_strides, units, dropout_rate, learning_rate, batch_size, epochs, custom_activation, custom_loss, input_shape)

    return best_model


def CNN_train_model(model, train_inputs, train_labels, config):
    """
    Train a TensorFlow model using early stopping and model checkpoints.

    Args:
        model (tf.keras.Model): A compiled TensorFlow model.
        train_inputs (numpy.ndarray): Training input data.
        train_labels (numpy.ndarray): Training labels.
        config (dict): A dictionary containing model options.

    Returns:
        tuple: 
            History (tf.keras.callbacks.History): Object that records training metrics for each epoch.
            Model (tf.keras.Model): The trained TensorFlow model.
    """
    # Define directory for saving model checkpoints
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..")
    model_dir = os.path.join(output_dir, "Models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate a subfolder name based on the model configuration
    subfolder = generate_subfolder_name(config["Mice Train"], config["Human Train"], config["Mice Test"], config["Human Test"])

    # Define paths for saving model checkpoints
    rn50_model_path = os.path.join(model_dir, 'ResNet50', subfolder, f'best_model_rn50.h5')
    custom_model_path = os.path.join(model_dir, 'Custom CNN', subfolder, f'best_model_custom.h5')

    rn50_pickle_path = os.path.join(model_dir, 'ResNet50', subfolder)
    custom_pickle_path = os.path.join(model_dir, 'Custom CNN', subfolder)

    # Calculate the validation split index
    train_inputs_split, val_inputs_split, train_labels_split, val_labels_split = train_test_split(train_inputs, train_labels, test_size=config['validation_split'], 
                                                                                                  random_state=config['random_state'], stratify=train_labels)

    # Define callbacks for early stopping and model saving
    early_stopping = EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'])
    custom_model_saver = CustomModelSaver(custom_model_path, custom_pickle_path, 'Custom CNN', config, validation_data=(val_inputs_split, val_labels_split))
    rn50_model_saver = CustomModelSaver(rn50_model_path, rn50_pickle_path, 'ResNet50', config, validation_data=(val_inputs_split, val_labels_split))

    # Train the model and save the best model 
    if config['CNN'] == 'custom':

        # Perform a grid search over a specified parameter grid
        grid = GridSearchCV(estimator=model, param_grid=config['params'], n_jobs=-1, cv=config['cv'])
        grid_result = grid.fit(train_inputs_split, train_labels_split, validation_data=(val_inputs_split, val_labels_split), callbacks=[early_stopping])
        
        # Update config with the best parameters found by the grid search
        config.update(grid_result.best_params_)

        # Build the final model with the best parameters
        best_model = create_best_model(config)

        datagen = ImageDataGenerator(**config['datagen_params'])

        # Fit the data generator to the training data
        datagen.fit(train_inputs_split)

        # Create a validation generator
        validation_datagen = ImageDataGenerator()
        validation_generator = validation_datagen.flow(val_inputs_split, val_labels_split, batch_size=config['batch_size'])

        # Fit the model with augmented data and validation set
        history = best_model.fit(datagen.flow(train_inputs_split, train_labels_split, batch_size=config['batch_size']),
                                validation_data=validation_generator,
                                epochs=config['epochs'],
                                steps_per_epoch=train_inputs_split.shape[0] / config['batch_size'],
                                validation_steps=val_inputs_split.shape[0] / config['batch_size'],
                                callbacks=[early_stopping, custom_model_saver])

    else:
        # Use the RESNET50 models
        best_model = model
        
        # Fit the model with validation split        
        history = best_model.fit(train_inputs_split, train_labels_split, 
                                 epochs=config['epochs'], 
                                 batch_size=config['batch_size'], 
                                 validation_data=(val_inputs_split, val_labels_split), 
                                 callbacks=[early_stopping, rn50_model_saver])  

    return history, best_model