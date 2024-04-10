

# Deep Learning; CNN

## Deep Learning Part A

This repository contains code for a deep learning model designed for image classification tasks. The model architecture is coded from scratch with customizable hyperparameters. It includes options for data augmentation, various activation functions, dropout, and more.

### Features:
- Customizable hyperparameters via command-line arguments
- Support for data augmentation during training
- Various activation functions including ReLU, GELU, SiLU, and Mish
- Dropout regularization for controlling overfitting
- Integration with Weights & Biases (WandB) for experiment tracking

### Dependencies:
- Python 3.x
- PyTorch
- torchvision
- WandB
- Matplotlib
- NumPy

### Setup:
1. Clone the repository:
    ```
    https://github.com/Abdul-Ansar/CS6910_Assignment_2.git
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

### Usage:
- To run the model with default hyperparameters:
    ```
    python main.py
    ```

- To specify hyperparameters via command-line arguments:
    ```
    python main.py --wandb_project <wandb_project_name> --wandb_entity <wandb_entity> --epochs <num_epochs> --num_filters <num_filters> --learning_rate <learning_rate> --activ_func <activation_function> --dropout_factor <dropout_factor> --filter_factor <filter_factor>
    ```

### Command-line arguments:
- `-wp` or `--wandb_project`: WandB project name for experiment tracking.
- `-we` or `--wandb_entity`: WandB entity for experiment tracking.
- `-e` or `--epochs`: Number of epochs for training the neural network.
- `-nf` or `--num_filters`: Number of filters in the convolutional neural network.
- `-lr` or `--learning_rate`: Learning rate used to optimize model parameters.
- `-af` or `--activ_func`: Activation function to use (options: ReLU, GELU, SiLU, Mish).
- `-df` or `--dropout_factor`: Dropout factor in the CNN.
- `-ff` or `--filter_factor`: Filter factor to adjust the number of filters in each layer.

### Results Tracking:
- Experiment results are tracked using WandB. You can view experiment metrics and visualizations in the WandB dashboard.


## Deep Learning Part B: Transfer Learning with VGG16

This repository contains a Python script for training an image classification model using the VGG16 architecture. The script allows you to specify various hyperparameters through command-line arguments, making it easy to experiment with different configurations.

### Requirements:
- Python 3.x
- PyTorch
- torchvision
- WandB (Weights & Biases) - for experiment tracking (optional)

### Installation:
1. Clone this repository to your local machine:
    ```
    git clone https://github.com/your_username/image-classification-vgg16.git
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

### Usage:

#### Training the Model:
You can train the image classification model using the provided `train.py` script. This script accepts various command-line arguments for specifying hyperparameters.
    ```
    python train.py --wandb_project your_project --wandb_entity your_entity --epochs 20 --num_filters 64 --learning_rate 0.001 --dropout_factor 0.3 --filter_factor 1
    ```

#### Directory Structure:
Make sure your dataset is organized in the following directory structure:
```
Tl with vgg16/
│
├── nature_12k/
│   ├── inaturalist_12K/
│   │   ├── train/
│   │   │   ├── class1/
│   │   │   │   ├── image1.jpg
│   │   │   │   ├── image2.jpg
│   │   │   │   └── ...
│   │   │   ├── class2/
│   │   │   │   ├── image1.jpg
│   │   │   │   ├── image2.jpg
│   │   │   │   └── ...
│   │   └── val/
│   │       ├── class1/
│   │       │   ├── image1.jpg
│   │       │   ├── image2.jpg
│   │       │   └── ...
│   │       ├── class2/
│   │       │   ├── image1.jpg
│   │       │   ├── image2.jpg
│   │       │   └── ...
│   └── ...
│
├── train.py
├── README.md
└── requirements.txt
```
### Author:
[Abdul-Ansar_GE23m018](https://github.com/Abdul-Ansar/)
