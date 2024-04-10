**Deep Learning Part A**

This repository contains code for a deep learning model designed for image classification tasks. The model architecture coded from scrach with customizable hyperparameters. It includes options for data augmentation, various activation functions, dropout, and more.

**Features:**
- Customizable hyperparameters via command-line arguments
- Support for data augmentation during training
- Various activation functions including ReLU, GELU, SiLU, and Mish
- Dropout regularization for controlling overfitting
- Integration with Weights & Biases (WandB) for experiment tracking

**Dependencies:**
- Python 3.x
- PyTorch
- torchvision
- WandB
- Matplotlib
- NumPy

**Setup:**
1. Clone the repository:

   ```
   https://github.com/Abdul-Ansar/CS6910_Assignment_2.git
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

**Usage:**
- To run the model with default hyperparameters:
  ```
  python main.py
  ```

- To specify hyperparameters via command-line arguments:
  ```
  python main.py --wandb_project <wandb_project_name> --wandb_entity <wandb_entity> --epochs <num_epochs> --num_filters <num_filters> --learning_rate <learning_rate> --activ_func <activation_function> --dropout_factor <dropout_factor> --filter_factor <filter_factor>
  ```

**Command-line arguments:**
- `-wp` or `--wandb_project`: WandB project name for experiment tracking.
- `-we` or `--wandb_entity`: WandB entity for experiment tracking.
- `-e` or `--epochs`: Number of epochs for training the neural network.
- `-nf` or `--num_filters`: Number of filters in the convolutional neural network.
- `-lr` or `--learning_rate`: Learning rate used to optimize model parameters.
- `-af` or `--activ_func`: Activation function to use (options: ReLU, GELU, SiLU, Mish).
- `-df` or `--dropout_factor`: Dropout factor in the CNN.
- `-ff` or `--filter_factor`: Filter factor to adjust the number of filters in each layer.

**Results Tracking:**
- Experiment results are tracked using WandB. You can view experiment metrics and visualizations in the WandB dashboard.

**Author:**
[Abdul Ahad](https://github.com/Abdul-Ansar/)
