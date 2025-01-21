import streamlit as st
from PIL import Image
import io
import os
import shutil
import model_utils
from model_utils import load_yolo_model

MODEL_NAME = "cockroach_detection.pt"
IMAGE_ADDRESS = "https://i.ytimg.com/vi/bEwCA_nrY5Q/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLB4y6YJGxw6oVxFm-Uucbrjfqhu2Q"
RADIO_IMAGE = "Image"
RADIO_VIDEO = "Video"
USER_IMAGE_NAME = "user_input.png"
USER_VIDEO_NAME = "user_video.avi"
PREDICTION_PATH = "runs/detect/predict"
PREDICTION_NEW_PATH = "predictions"
PREDICTION_IMAGE_PATH = PREDICTION_NEW_PATH + "/" + USER_IMAGE_NAME
PREDICTION_VIDEO_PATH = PREDICTION_NEW_PATH + "/" + USER_VIDEO_NAME
FINAL_PREDICTION_VIDEO = "output.mp4"
VIDEO_EXTENSION = ".avi"

# Load the PyTorch weights
model = load_yolo_model(MODEL_NAME)

# Get predictions
def get_predictions(source_path):
    model.predict(source_path, save=True)
    print("Prediction Complete")

    # Copy the predictions and save them independently
    if os.path.exists(PREDICTION_PATH):
        for item in os.listdir(PREDICTION_PATH):
            shutil.copy(os.path.join(PREDICTION_PATH, item), os.path.join(PREDICTION_NEW_PATH, item))
            if item.endswith(VIDEO_EXTENSION):
                model_utils.convert_mp4_H264(os.path.join(PREDICTION_NEW_PATH, item), os.path.join(PREDICTION_NEW_PATH, FINAL_PREDICTION_VIDEO))
                print("Video Conversion Complete!!!!!!")
            os.remove(os.path.join(PREDICTION_PATH, item))

    os.rmdir(PREDICTION_PATH)
    print("Folder removed!")

    return True

# Web application
st.title("Cockroach Detection")

# Set an image
st.image(IMAGE_ADDRESS, caption="Cockroach Detection")

# Detection dashboard
st.header("Detection Dashboard 📷")

# Sidebar
with st.sidebar:
    results = False
    video_results = False
    user_image = None
    user_video = None

    # Set a header
    st.header("Cockroach Detection")

    # Choose a method
    st.subheader("Select a method")

    # Set a radio button
    option = st.radio("Select an input option", [RADIO_IMAGE, RADIO_VIDEO], captions=["Detect signs on Images", "Detect signs on videos"])

    # If image
    if option == RADIO_IMAGE:
        user_image = st.file_uploader("Upload an image", accept_multiple_files=False, help="Upload any image from your local", type=["png", "jpg", "jpeg"])

        if user_image:
            if st.button("Detect Signs", use_container_width=True, type='primary'):
                results = get_predictions(USER_IMAGE_NAME)

    if option == RADIO_VIDEO:
        user_video = st.file_uploader("Upload a video", accept_multiple_files=False, help="Upload any video from your local", type=["mp4", "avi", "mpeg"])
        if user_video:
            if st.button("Detect Signs", use_container_width=True, type='primary'):
                with st.spinner("Processing the Video...."):
                    video_results = get_predictions(USER_VIDEO_NAME)

# Create two columns
col1, col2 = st.columns(2)

if user_image:
    with col1:
        st.subheader("User Input")
        # Set the user image
        st.image(user_image)

        # Read and save the image
        image_bytes = io.BytesIO(user_image.read())
        input_image = Image.open(image_bytes)
        input_image.save(USER_IMAGE_NAME)

if results:
    with col2:
        st.subheader("Prediction")
        # Set the user image
        st.image(PREDICTION_IMAGE_PATH)

if user_video:
    # Set the user video
    st.subheader("User Input Video")
    st.video(user_video)

    # Read bytes
    video_bytes = io.BytesIO(user_video.read())

    with open(USER_VIDEO_NAME, "wb") as video_file:
        video_file.write(video_bytes.read())

    video_file.close()

if video_results:
    st.subheader("Predicted Video")
    st.video(os.path.join(PREDICTION_NEW_PATH, FINAL_PREDICTION_VIDEO))
