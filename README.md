# SenID: Cell Senescence Classification

This project aims to develop a robust and accurate classification system to identify cell senescence using deep learning techniques. Cellular senescence is a natural process of cell aging that plays a critical role in various biological processes such as tumor suppression, tissue repair, and aging. By identifying senescent cells, researchers can better understand the underlying mechanisms and develop targeted therapies for age-related diseases and cancer.

Our approach for this project involves utilizing state-of-the-art Convolutional Neural Networks (CNNs) to learn and classify various morphological features present in the cell images. The modular design of the code allows for easy customization and adaptability to different datasets and model architectures.

The code repository is organized into separate modules for each task, such as data preprocessing, model building, training, and visualization. This enables users to easily modify and experiment with different components of the project.

Key features of this project include:

1. A comprehensive data preprocessing pipeline to prepare cell images for classification tasks.
2. Implementation of two CNN (ResNet50 and a custom CNN) architectures for model building and fine-tuning.
3. Efficient training procedures with support for early stopping, learning rate scheduling, and model checkpointing.
4. Visualization tools to analyze and interpret model performance, feature importance, and classification results.

In this project, we utilize both human and mouse cell data collected in-house, which allows for the development of a more diverse and versatile classification system that can be applied to a wide range of cell types and organisms.

We currently have two branches, Main and DNN. Main contains functioning and working code, while DNN contains work-in-progress DNN architecture. When DNN is finished, the branches will be merged.

* If you are interested in delving deeper into the science behind our work or want a more thorough overview of our process, please refer to our presentation or written report located in the 'reports' folder.

## Table of Contents
- [Getting started](#getting-started)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Contributing](#contributing)
- [Further Improvements](#further-improvements)

## Getting Started

To get started with the SenID project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies (listed below) through the correct yml file.
3. Execute the data preprocessing pipeline to prepare your cell images for classification.
4. Train and evaluate various CNN models using the provided scripts.
5. Visualize and interpret the results using the provided visualization tools.

## Requirements

The following packages are required:

1. cellpose
2. keras/tensorflow
3. lime
4. matplotlib
5. numpy
6. opencv
7. pandas
8. Pillow
9. scikit_image
10. scikit_learn

Please refer to the requirements.txt file (inside requirements folder) for the specific versions of these packages that are required.

Additionally, check the SenID.yml files (inside requirements folder) for more specific configurations compatible with this project. There are two versions of this file: one for Windows (with GPU support) and another for Mac (supporting M1 chip).

## Installation

To install the required packages, run:

```bash
1. conda env create -f /path/to/your/environment.yml
2. conda activate your_environment_name
```

## Usage

1. Run the main script:

```bash
python main.py
```

## Code Overview
- `main.py`: The main script that runs the entire process of loading data, training, and evaluating the model. It also handles the combination of different datasets (human and mice) and the selection of CNN models for training.
- `preprocess.py`: Contains functions for loading, resizing, and cropping images and generating DNN feature data - only in DNN branch.
- `CNN_Model.py`: Contains functions for loading and processing images, building and training CNN models based on custom architecture or ResNet50.
- `utils.py`: Contains utility functions for generating subfolder names, making predictions, evaluating models, further cropping, and explaining model predictions using LIME.
- `visuals.py`: Contains functions for visualizing the model's training history, creating confusion matrices, plotting ROC curves, plotting precision-recall curves, and displaying LIME explanations.
- `DNN_Model.py`:  Contains functions for loading and processing images-to-tabular data and building and training DNN model based on custom architecture. Work in progress and currently only in DNN branch

## Contributing

We encourage and welcome contributions from the community. If you'd like to contribute to the SenID project, please feel free to fork the repository, create a new branch, and submit a pull request with your proposed changes.

## Further Improvements

We are currently working on creating a feature-based DNN and combining this with our pre-trained ResNet50 CNN to create a multi-model CNN. When finished, it will be updated along with the rest of the code. 
