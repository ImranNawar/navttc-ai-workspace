"""
Facial Emotion Recognition System
This file can be used for inference using the pretrained model, which incorporated CLIP encoder 
to extract features and then classifies facial emotions using a fully connected layer. This model 
achieved about 62% accuracy on the test data.
"""
import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
from model_architecture import CLIPClassifier
import time

st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")

st.sidebar.header("Facial Emotion Recognition System")
st.sidebar.write("This application uses a CLIP model fine-tuned on the FER-2013 dataset to perform real-time emotion recognition from a live webcam feed.")

PRETRAINED_MODEL = 'models/clip_fer.pth'

def load_model(model_path, num_classes=7):
    """This model loads the custom defined FERModel.

    Args:
        model_path (str): path to the pretrained model
        num_classes (int, optional): number of classes in the dataset.
        defaults to 7.

    Returns:
        FERModel: Custom FERModel
    """
    model = CLIPClassifier(num_classes)
    state_dict = torch.load(model_path,
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image):
    """Preprocess an image so that it can be used by the FERModel for inference

    Args:
        image (ndarray): a grayscale image

    Returns:
        tensor: preprocessed image
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0)

# Load the pretrained model
model = load_model(PRETRAINED_MODEL)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Streamlit app layout
st.title("Facial Emotion Recognition System")

# Variables to manage frame processing and performance
cap = None
frame_count = 0
fps = 0
start_time = time.time()
inference_times = []

# Start webcam if "Open Webcam" is pressed
if cap is None and st.button("Open Webcam"):
    cap = cv2.VideoCapture(0)  # Open webcam
    video_placeholder = st.empty()
    emo_placeholder = st.empty()

    # Display "Close Webcam" button only when webcam is active
    close_webcam = st.button("Close Webcam")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("No video feed detected. Please check your camera.")
            break

        # Process frame and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)
        inference_start = time.time()

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_tensor = preprocess_image(face)

            with torch.no_grad():
                output = model(face_tensor)
                _, predicted = torch.max(output, 1)
                emotion = emotions[predicted.item()]

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (210, 140, 70), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 200, 0), 2)

        # Check if "Close Webcam" is pressed
        if close_webcam:
            cap.release()
            video_placeholder.empty()
            emo_placeholder.empty()
            st.write("Webcam feed stopped.")
            break

        # Calculate FPS and inference time
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        frame_count += 1
        if (time.time() - start_time) > 1.0:
            fps = frame_count / (time.time() - start_time)
            frame_count = 0
            start_time = time.time()

        # Display on Streamlit app
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
else:
    st.info("Click the 'Open Webcam' button to start facial emotion recognition.")

# Footer
footer = """
<div style="position: fixed; bottom: 0; width: 100%; background-color: #EDF3FA; padding: 10px; text-align: center;">
    Created by Imran Nawar
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
