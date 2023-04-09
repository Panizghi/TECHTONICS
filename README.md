# TECHTONICS
This repository contains the code and information for building a Convolutional Neural Network (CNN) model for detecting earthquakes using the Stanford Earthquake Dataset (STEAD). The model is implemented in Python using the Keras framework.

Overview
The repository contains the following files:
-
-
-

STEAD Dataset
The STEAD dataset is a global dataset of seismic signals that can be used for training AI models for earthquake detection. The dataset contains over 13,000 seismograms from 58 different earthquakes around the world. Each seismogram is a 3-component, 4-second waveform recorded by a seismograph station. The dataset is available for download: https://github.com/smousavi05/STEAD

CNN Model
The CNN model implemented in this repository takes the seismograms from the STEAD dataset as input and classifies them as either containing an earthquake or not. The model architecture consists of three 1D convolutional layers, followed by two fully connected layers. The output layer has a single sigmoid unit for binary classification. The model was trained using the Adam optimizer and binary cross-entropy loss function. We have also used data augmentation techniques such as random shifts and flips to increase the size of the training dataset and improve the model's generalization performance.
