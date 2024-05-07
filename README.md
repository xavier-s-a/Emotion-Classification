# Emotion Recognition System

## Overview
This project is designed to detect human emotions from images and videos using a deep learning model. The model identifies emotions from facial expressions and is implemented using PyTorch.

## Components

### Deep Learning Model - EmotionNet
`EmotionNet` is a convolutional neural network designed to classify seven different emotions from facial images. It is built using PyTorch and trained on the FER-2013 dataset.

### Data Processing and Augmentation
Transforms include resizing, normalization, random flips, and rotations to make the model robust to various facial orientations and lighting conditions.

### Training and Validation
train.py is used to train and validate the model over multiple epochs, displaying training and validation losses and accuracies, and plotting confusion matrices to visualize model performance.

### Real-time Emotion Recognition
detect.py utilizes MTCNN for face detection and the trained EmotionNet model for emotion classification. It supports processing single images, video files, and live webcam feeds.

## Installation
To run this project, you need Python 3.x and the following libraries:
- PyTorch
- torchvision
- OpenCV
- facenet_pytorch
- numpy
- matplotlib
- seaborn
- scikit-learn
- PIL
- tkinter

You can install the required libraries using pip:
```bash
pip install -r requirements.txt

```
## Usage
### Testing the System in browser
Run the following command to use the pre-trained model for detecting emotions in images or videos:
```bash
python app.py
```

### Testing the System with CLI

Run the following command to use the pre-trained model for detecting emotions in images or videos:
```bash
python detect.py
```
### Training the Model

To train the model from scratch:
```bash
python train.py 
```

## Features
- **Emotion Detection from Images:** Load images and detect emotions from faces using a trained neural network model.
- **Emotion Detection from Videos:** Process videos to detect and label emotions frame by frame.
- **Webcam Support:** Real-time emotion detection using a webcam.



[URL for Fer2013 data set]([<https://duckduckgo.com/?q=test this!&t=ffab>](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data))

