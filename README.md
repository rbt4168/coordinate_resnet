# Coordinate ResNet

## Overview

The Coordinate ResNet is a modified ResNet18 architecture that incorporates spatial information using a custom SpatialSoftmax layer. The model calculates the expected 2D coordinate for each channel, enhancing its ability to capture spatial relationships in the data.

## Components

### 1. SpatialSoftmax

The `SpatialSoftmax` module is a custom PyTorch layer that applies spatial softmax along the height and width dimensions of an input tensor. It computes the expected 2D coordinate for each channel, enabling the model to capture spatial information effectively.

### 2. SpatialResNet18

The `SpatialResNet18` model is built on top of the ResNet18 architecture, integrating the `SpatialSoftmax` layer. This modification replaces the average pooling layer with spatial softmax and adapts the fully connected layer for the desired output dimension.

### 3. ResNet18

The standard `ResNet18` model is also provided without the spatial softmax layer. It serves as a baseline ResNet18 architecture for comparison.

## main.py
The provided script generates synthetic test data for image processing tasks. It creates images, each with Gaussian noise and two randomly placed points of interest. The images are saved with a recognizable filename in the "pictures" directory, and the coordinates of the two points for each image are stored in a JSON file named "coordinates.json." This synthetic dataset is useful for testing and validating models that analyze spatial information in images.

## Usage

### Getting the Model

Use the `get_model` function to obtain an instance of the Coordinate ResNet.

```python
from coordinate_resnet import get_model

# Get the Coordinate ResNet model
model = get_model("spatial_resnet18", pretrained=True)
