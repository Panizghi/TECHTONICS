# Earthquake Detection with Convolutional Neural Networks (CNN)

This repository contains the code and information for building Convolutional Neural Network (CNN) models for earthquake detection using the Stanford Earthquake Dataset (STEAD). The models are implemented in Python using the Keras framework.

## Overview

CNN classification is employed for the prediction and classification of seismic signals into earthquakes and noise. Additionally, a CNN regression model is used for analyzing the characteristics of the seismic waves. Both models utilize callbacks to save the partially-trained model at the end of each epoch.

### High-level Procedure:

1. Randomly chooses desired-size images.
2. Performs a train-test split.
3. Compiles and fits the CNN model.
4. Evaluates and saves the model.
5. Produces evaluation figures.

## STEAD Dataset

The STEAD dataset is a global collection of seismic signals suitable for training AI models for earthquake detection. It includes over 13,000 seismograms from 58 different earthquakes worldwide. Each seismogram is a 3-component, 4-second waveform recorded by a seismograph station. The dataset is available for download [here](https://github.com/smousavi05/STEAD). Our data is a subset of STEAD, comprising 71,500 noise samples and 196,322 earthquake signals (approximately 36% noise and 64% earthquakes), resulting in a total of 267,822 dataset samples.

## Models and Layers

### Classification CNN:

- **Conv2D:** Applies convolution operation on the input image. 32 filters with a (5,5) kernel size and ReLU activation function.
- **MaxPool2D:** Performs max pooling operation, reducing spatial dimensions.
- **Dropout:** Randomly drops out input units to prevent overfitting.
- **Flatten:** Flattens the output into a one-dimensional vector.
- **Dense Layers:** Fully connected layers with ReLU activation. Output layer has 2 units with softmax activation for final class probabilities.

### Regression CNN:

- **Conv2D:** Applies convolution operation with 64 filters, (5,5) kernel size, and ReLU activation.
- **MaxPool2D:** Performs max pooling operation, reducing spatial dimensions.
- **Dropout:** Randomly drops out input units to prevent overfitting.
- **Flatten:** Flattens the output into a one-dimensional vector.
- **Dense Layers:** Fully connected layers with ReLU activation. Output layer has 1 unit, predicting the value for each input image.

## Loss and Optimizer

### Classification CNN:

- **Optimizer:** Adam, utilizing gradients of previous time steps to calculate the current step's learning rate. Well-suited for sparse data and noisy gradients.
- **Loss Function:** Categorical cross-entropy.

### Regression CNN:

- **Optimizer:** Adam with a learning rate of 1e-5.
- **Loss Function:** Mean Squared Error (MSE). Penalizes the model more heavily for larger errors, commonly used for regression problems.

Feel free to use and modify the code for your earthquake detection projects! If you have any questions or feedback, please don't hesitate to reach out.
