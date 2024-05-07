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



## Team Members

- **Amith:** Focused on face detection module development, testing the system on images, and integrating the model with video and webcam functionalities.
- **Xavier:** Responsible for training the neural network and developing the Graphical User Interface (GUI).


## Team Contributions
**Xavier** 

### Responsibilities:

- Neural Network Development: Designed and implemented the EmotionNet neural network using PyTorch. 
- GUI Development: Developed the  graphical user interface using Tkinter. This interface allows users to upload images and videos, and to interact with the system in real-time via a webcam.
- Project Integration: Integration of different components of the project, making the transition between the neural network processing and the user interface smooth.

**Amith**

### Responsibilities:

- Face Detection Module: Implemented the face detection functionality using the MTCNN model. Ensuring that the system could identify and process faces accurately from various inputs.
- System Testing: Conducted extensive testing of the emotion recognition capabilities on static images, ensuring robust performance.
- Integration with Real-Time Systems: Responsible for integrating the face detection system with real-time video and webcam feeds, allowing the system to perform emotion detection live.
