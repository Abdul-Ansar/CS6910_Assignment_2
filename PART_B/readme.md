

# Deep Learing Part B; Transfer Learning  with VGG16

This repository contains a Python script for training an image classification model using the VGG16 architecture. The script allows you to specify various hyperparameters through command-line arguments, making it easy to experiment with different configurations.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- WandB (Weights & Biases) - for experiment tracking (optional)

## Installation

1. Clone this repository to your local machine:

    ```bash
   https://github.com/Abdul-Ansar/CS6910_Assignment_2.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

You can train the image classification model using the provided `train.py` script. This script accepts various command-line arguments for specifying hyperparameters.

```bash
python train.py --wandb_project your_project --wandb_entity your_entity --epochs 20 --num_filters 64 --learning_rate 0.001 --dropout_factor 0.3 --filter_factor 1
```

- `--wandb_project`: Name of your WandB project (optional, default: "de2_ge23m18").
- `--wandb_entity`: WandB entity used to track experiments (optional, default: "ge23m18").
- `--epochs`: Number of epochs to train the neural network (default: 1).
- `--num_filters`: Number of filters in the convolutional neural network (default: 3).
- `--learning_rate`: Learning rate used to optimize model parameters (default: 0.001).
- `--dropout_factor`: Dropout factor in the CNN (default: 0.3).
- `--filter_factor`: Filter factor (default: 1).

### Directory Structure

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
