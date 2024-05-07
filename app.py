from flask import Flask, request, render_template, Response, redirect, url_for, session,jsonify
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
from EmotionNet import EmotionNet
from facenet_pytorch import MTCNN
import os
import tempfile
import base64
import io
import numpy as np

app = Flask(__name__)
device = torch.device("cpu")
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)
# Load model
model = EmotionNet()
model.load_state_dict(torch.load('/Users/xavier/Desktop/CS712/finalproject/weights_emotions.pth', map_location=torch.device('cpu')))
model.eval()
mtcnn = MTCNN(keep_all=True, device=device)
def detect_and_recognize(image,save_path=None):
    try:
        # Convert the image to RGB from BGR, which OpenCV uses
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Detect faces in the image
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
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    annotated_image = detect_and_recognize(img_np)
    img_bytes = io.BytesIO()
    Image.fromarray(annotated_image).save(img_bytes, format='JPEG')
    encoded_img_data = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    return jsonify({'image': encoded_img_data})



@app.route('/predict_video', methods=['POST'])
def predict_video():
    file = request.files['media']
    if not file:
        return 'No file uploaded.', 400
    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    video_file_path = os.path.join(temp_dir, file.filename)
    file.save(video_file_path)
    video = cv2.VideoCapture(video_file_path)
    response = Response(gen_frames(video), mimetype='multipart/x-mixed-replace; boundary=frame')
    # Clean up the temporary file after use
    os.remove(video_file_path)
    os.rmdir(temp_dir)
    return response

@app.route('/webcam')
def webcam():
    return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(video):
    while True:
        success, frame = video.read()
        if not success:
            break
        frame = detect_and_recognize(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_webcam():
    cap = cv2.VideoCapture(1)  # Ensure this is the correct camera index
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            processed_frame = detect_and_recognize(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                print("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

if __name__ == '__main__':
    app.run(debug=True)
