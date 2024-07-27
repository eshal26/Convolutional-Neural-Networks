# Custom Convolutional Neural Network Models

This repository contains custom implementations of convolutional neural network (CNN) models designed for binary classification tasks. Each model has been tailored with specific architectures to enhance performance and adapt to various classification challenges.

## Models Overview

### Simple Inception Model
- **Architecture**: Custom inception-like model with basic components.
- **Key Components**:
  - **Initial Convolution**: `Conv2d(3, 16, kernel_size=3, stride=1, padding=1)`
  - **Inception Block**:
    - **Branch 1**: `Conv2d(16, 16, kernel_size=3, stride=1, padding=1)`, followed by `ReLU`
    - **Branch 2**: `MaxPool2d(kernel_size=3, stride=1, padding=1)` followed by `Conv2d(16, 16, kernel_size=1, stride=1)` and `ReLU`
  - **Dropout Layers**: `Dropout(p=0.3)` applied twice
  - **Flatten Layer**: Converts tensor to 1D
  - **Fully Connected Layer**: `Linear(in_features=1605632, out_features=2)`

### Simple ResNet Model
- **Architecture**: Custom ResNet with basic residual blocks.
- **Key Components**:
  - **Residual Block**:
    - **Convolutions**: `Conv2d(3, 64, kernel_size=3, padding=1)`, `Conv2d(64, 64, kernel_size=3, padding=1)`
    - **Batch Normalization**: Applied after convolutions
    - **ReLU Activation**: Applied after normalization
    - **Shortcut Connection**: Adjusted dimensions with `Conv2d(3, 64, kernel_size=1, stride=1)` if needed
  - **Dropout Layers**: `Dropout(p=0.5)` applied twice
  - **Pooling**: `AdaptiveAvgPool2d(output_size=(1, 1))`
  - **Fully Connected Layer**: `Linear(in_features=64, out_features=2)`

### Custom VGG Model
- **Architecture**: Custom VGG-like model.
- **Key Components**:
  - **Convolutions**: 
    - `Conv2d(3, 64, kernel_size=3, stride=1, padding=1)`, followed by `ReLU` and `MaxPool2d(kernel_size=2, stride=2)`
    - `Conv2d(64, 128, kernel_size=3, stride=1, padding=1)`, followed by `ReLU` and `MaxPool2d(kernel_size=2, stride=2)`
  - **Dropout Layers**: `Dropout(p=0.5)` applied twice
  - **Flatten Layer**: Converts tensor to 1D
  - **Fully Connected Layer**: `Linear(in_features=401408, out_features=2)`

### Simplified Xception Model
- **Architecture**: Simplified version of Xception.
- **Key Components**:
  - **Depthwise Separable Convolution**:
    - **Depthwise Convolution**: `Conv2d(3, 3, kernel_size=3, stride=2, padding=1, groups=3)`
    - **Pointwise Convolution**: `Conv2d(3, 64, kernel_size=1, stride=1)`
  - **Activation and Normalization**: `ReLU` followed by `BatchNorm2d(64)`
  - **Pooling**: `MaxPool2d(kernel_size=3, stride=2, padding=1)`
  - **Dropout Layers**: `Dropout(p=0.5)` applied twice
  - **Fully Connected Layer**: `Linear(in_features=200704, out_features=2)`

## Contribution
Feel free to fork the repository, make modifications, and submit pull requests. Contributions and suggestions are welcome.
