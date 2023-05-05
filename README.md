# TECHTONICS
This repository contains the code and information for building a Convolutional Neural Network (CNN) model for detecting earthquakes using the Stanford Earthquake Dataset (STEAD). The model is implemented in Python using the Keras framework.

## Overview
CNN classification has been used for prediction and classification of signals to earthquake vs noise 
CNN regression model has been used for analyzing characteristics of the wave
Both model uses callbacks to save the partially-trained model at the end of each epoch
### High-level procedure : 
- Chooses desired size images randomly 
- Performs a train-test split
- Compiles and then fits cnn model
- Evaluates and saves the model 
- Produces evaluation figures 


## STEAD Dataset
The STEAD dataset is a global dataset of seismic signals that can be used for training AI models for earthquake detection. The dataset contains over 13,000 seismograms from 58 different earthquakes around the world. Each seismogram is a 3-component, 4-second waveform recorded by a seismograph station. The dataset is available for download: https://github.com/smousavi05/STEAD
Our data is subset of STEAD with 71,500 noise samples to 196,322 earthquake signals ( about 36% noise and 64% earthquakes) for total of 267,822 total dataset samples.

## Models
### Layers
#### Classification CNN:
- Conv2D: This layer applies convolution operation on the input image. It has 32 filters with a kernel size of (5,5), which means that the filters are 5x5 in size. The activation function used is ReLU.
- MaxPool2D: This layer performs max pooling operation on the output of the previous convolutional layer. It has a pool size of (2,2), which means that it takes the maximum value of a 2x2 window and outputs it. This layer reduces the spatial dimensions of the output, making the model more computationally efficient.
- Dropout: This layer randomly drops out some of the input units, which helps prevent overfitting of the model.
- Flatten: This layer flattens the output of the previous layer into a one-dimensional vector. This is necessary so that the output can be fed into the subsequent dense layers.
- Dense: This layer is fully connected and has 64 units with a ReLU activation function.
- Dense: This layer is also fully connected and has 16 units with a ReLU activation function.
- Dense: This layer is fully connected and has 2 units with a softmax activation function. This layer outputs the final class probabilities for each input image.

#### Regression CNN:
- Conv2D: This layer applies convolution operation on the input image. It has 64 filters with a kernel size of (5,5), which means that the filters are 5x5 in size. The activation function used is ReLU.
- MaxPool2D: This layer performs max pooling operation on the output of the previous convolutional layer. It has a pool size of (2,2), which means that it takes the maximum value of a 2x2 window and outputs it. This layer reduces the spatial dimensions of the output, making the model more computationally efficient.
- Dropout: This layer randomly drops out some of the input units, which helps prevent overfitting of the model.
- Flatten: This layer flattens the output of the previous layer into a one-dimensional vector. This is necessary so that the output can be fed into the subsequent dense layers.
- Dense: This layer is fully connected and has 16 units with a ReLU activation function.
- Dense: This layer is also fully connected and has 1 unit, which outputs the predicted value for each input image. The activation function is not specified, which means that it defaults to a linear activation function.

### Loss 
#### Classification CNN:
the optimizer used is Adam. It uses the gradients of the previous time steps to calculate the current step's learning rate, which makes it well suited for sparse data and noisy gradients. 
The loss function used in the classification CNN is categorical cross-entropy.
#### Regression CNN: 
the optimizer used is also Adam with a learning rate of 1e-5. The loss function used in the regression CNN is mean squared error (MSE). MSE is a commonly used loss function for regression problems that calculates the average of the squared differences between the predicted and true values. It is a measure of the model's performance and penalizes the model more heavily for larger errors.


