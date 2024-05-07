import tkinter.simpledialog as simpledialog
from PIL import Image, ImageTk
import threading
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
from EmotionNet import EmotionNet

device = torch.device("cpu")

model = EmotionNet().to(device)
#if a user needs another trained weight change here
model.load_state_dict(torch.load('/Users/xavier/Desktop/CS712/finalproject/weights_emotions.pth'))
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
def detect_and_recognize(image,save_path=None):
    try:
        # Convert the image to RGB from BGR, (for OpenCV uses)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Detect faces in the image,MTCNN
        boxes, _ = mtcnn.detect(image_pil)

        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Clamp coordinates and validate rectangle
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)
                if x1 >= x2 or y1 >= y2:
                    continue  # Skip invalid boxes

                face = image_rgb[y1:y2, x1:x2]
                face_pil = Image.fromarray(face)
                tensor = preprocess_image(face_pil)
                tensor = tensor.to(device)
                outputs = model(tensor)
                _, predicted = torch.max(outputs, 1)
                emotion = emotions[predicted.item()]

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if save_path:
            cv2.imwrite(save_path, image)

    except Exception as e:
        print(f"Error processing frame: {e}")

    return image

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        save_path = file_path.rsplit('.', 1)[0] + '_annotated.jpg'
        image = detect_and_recognize(image,save_path=save_path)
        show_image(image)

def show_image(image):
    # Convert the image to a format that Tkinter can use
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.photo = image  
    image_label.configure(image=image)
    root.geometry(f'{image.width()}x{image.height()}+100+100') 


def threaded_video_play(cap):
    thread = threading.Thread(target=play_video, args=(cap,))
    thread.start()

def open_video():
    video_path = filedialog.askopenfilename()
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            threaded_video_play(cap)
        else:
            print("Failed to open video.")

def play_video(cap):
    import random
    a=random.randint(0,100)
    save_path = 'result'+str(a)+'.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret, frame = cap.read() 
    if not ret:
        print("Failed to get the video frame.")
        return
    
    output_video = cv2.VideoWriter(save_path, fourcc, 24, (frame.shape[1], frame.shape[0]))

    while ret:
        frame = detect_and_recognize(frame)
        show_image(frame)  
        output_video.write(frame)  

        ret, frame = cap.read()  
        if cv2.waitKey(1) == 27:  # ESC key to stop
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    print("Video processing complete and saved to:", save_path)





root = tk.Tk()
root.title("Emotion Recognition")

image_label = tk.Label(root)
image_label.pack()

        
def update_image():
    global cap
    ret, frame = cap.read()
    if ret:
        frame = detect_and_recognize(frame)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)
    image_label.after(10, update_image)

def webcam_thread():
    global cap
    cap = cv2.VideoCapture(1)  # Index 0 for the default camera
    update_image()

def open_webcam():
    threading.Thread(target=webcam_thread).start()

def center_window(width, height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))
    root.geometry(f'{width}x{height}+{x}+{y}')

    
    


# Buttons for different functionalities
btn_load_image = tk.Button(root, text="Load Image", command=open_image)
btn_load_image.pack(side=tk.LEFT)

btn_load_video = tk.Button(root, text="Load Video", command=open_video)
btn_load_video.pack(side=tk.LEFT)

btn_webcam = tk.Button(root, text="Webcam", command=open_webcam)
btn_webcam.pack(side=tk.LEFT)

center_window(640, 480)

root.mainloop()