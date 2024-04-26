import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import torch
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
])

train_dataset = datasets.ImageFolder('/Users/xavier/Desktop/CS712/project/fer2013/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.ImageFolder('/Users/xavier/Desktop/CS712/project/fer2013/validation', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model, optimizer and loss function
model = EmotionNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training function
def train_and_validate(model, train_loader, test_loader, criterion, optimizer, num_epochs=70):
    model.train()

    # Lists to track performance metrics
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_correct / train_total)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.view(-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_losses.append(val_loss / len(test_loader))
        val_accs.append(val_correct / val_total)

        # Print accuracy on the test set
        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}')
        print(f'Accuracy of the network on the test images: {100 * val_correct / val_total:.2f}%')
    
    # Plotting training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return train_losses, train_accs, val_losses, val_accs

# Call the function with the model, data loaders, criterion, and optimizer
train_and_validate(model, train_loader, test_loader, criterion, optimizer, num_epochs=100) # need to increase num_epochs


torch.save(model.state_dict(), '/Users/xavier/Desktop/CS712/project/weights_emotions.pth')
model.load_state_dict(torch.load('/Users/xavier/Desktop/CS712/project/weights_emotions.pth'))



# Set device to CPU
device = torch.device('cpu')
model.to(device) 
model.eval()

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Function to preprocess the image for emotion detection
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# Function to detect faces and recognize emotions using MTCNN and EmotionNet
def detect_and_recognize(image):
    # Convert the image to RGB from BGR, which OpenCV uses
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Detect faces in the image
    boxes, _ = mtcnn.detect(image_pil)

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = image_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face)
            tensor = preprocess_image(face_pil)
            tensor = tensor.to(device)
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotions[predicted.item()]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image

def open_image():
    # Use file dialog to get the image path
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = detect_and_recognize(image)
        show_image(image)

def show_image(image):
    # Convert the image to a format that Tkinter can use
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.photo = image  # keep a reference to the image

def open_video():
    video_path = filedialog.askopenfilename()
    if video_path:
        cap = cv2.VideoCapture(video_path)
        play_video(cap)

def play_video(cap):
    ret, frame = cap.read()
    if ret:
        frame = detect_and_recognize(frame)
        show_image(frame)
        root.after(10, lambda: play_video(cap))  # Continue playing video
    else:
        cap.release()

def open_webcam():
    cap = cv2.VideoCapture(0)
    play_video(cap)

root = tk.Tk()
root.title("Emotion Recognition")

# Image label to display images
image_label = tk.Label(root)
image_label.pack()

# Buttons for different functionalities
btn_load_image = tk.Button(root, text="Load Image", command=open_image)
btn_load_image.pack(side=tk.LEFT)

btn_load_video = tk.Button(root, text="Load Video", command=open_video)
btn_load_video.pack(side=tk.LEFT)

btn_webcam = tk.Button(root, text="Webcam", command=open_webcam)
btn_webcam.pack(side=tk.LEFT)

root.mainloop()