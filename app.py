import streamlit as st
from PIL import Image
#from streamlit_webrtc import webrtc_streamer, WebRtcMode,ClientSettings
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)
import io
import av
import numpy as np
import os
import shutil
import model_utils
from model_utils import load_yolo_model
from turn import get_ice_servers


MODEL_NAME = "cockroach_detection.pt"
IMAGE_ADDRESS = "https://i.ytimg.com/vi/bEwCA_nrY5Q/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLB4y6YJGxw6oVxFm-Uucbrjfqhu2Q"
RADIO_IMAGE = "Image"
RADIO_VIDEO = "Video"
RADIO_WEBCAM = "Webcam"
USER_IMAGE_NAME = "user_input.png"
USER_VIDEO_NAME = "user_video.avi"
PREDICTION_PATH = "runs/detect/predict"
PREDICTION_NEW_PATH = "predictions"
PREDICTION_IMAGE_PATH = PREDICTION_NEW_PATH + "/" + USER_IMAGE_NAME
PREDICTION_VIDEO_PATH = PREDICTION_NEW_PATH + "/" + USER_VIDEO_NAME
FINAL_PREDICTION_VIDEO = "output.mp4"
VIDEO_EXTENSION = ".avi"


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

#load the pytorch weights
model = load_yolo_model(MODEL_NAME)

#get predictions
def get_predictions(source_path):
    model.predict(source_path, save = True)
    print("Prediction Complete")

    #copy the predictions and save it independently
    #this is mainly to avoid prediction tree structure
    if os.path.exists(PREDICTION_PATH):
        for item in os.listdir(PREDICTION_PATH):
            shutil.copy(os.path.join(PREDICTION_PATH, item), os.path.join(PREDICTION_NEW_PATH, item))
            if item.endswith(VIDEO_EXTENSION):
                model_utils.convert_mp4_H264(os.path.join(PREDICTION_NEW_PATH, item), os.path.join(PREDICTION_NEW_PATH, FINAL_PREDICTION_VIDEO))
                print("Video Conversion Complete!!!!!!")
            os.remove(os.path.join(PREDICTION_PATH, item))

    #then remove the predict director
    os.rmdir(PREDICTION_PATH)
    print("Folder removed!")

    return True


#web application
st.title("Cockroach Detection Detection")

#set an image
st.image(IMAGE_ADDRESS , caption = "Cockroach Detection")

#detection dashboard
st.header("Detection Dashboard 📷")


#sidebar
with st.sidebar:
    results = False
    video_results = False
    user_image = None
    user_video = None
    #set a header
    st.header("Cockroach Detection")

    #choose a method
    st.subheader("Select a method")

    #set a radio button
    option = st.radio("Select an input option", [RADIO_IMAGE , RADIO_VIDEO, RADIO_WEBCAM], captions=["Detect signs on Images", "Detect signs on videos", "Real time detection"])

    #if image
    if option == RADIO_IMAGE:
        user_image = st.file_uploader("Upload an image", accept_multiple_files=False, help = "Upload any image from your local", type = ["png", "jpg", "jpeg"])

        if user_image:
            if st.button("Detect Signs", use_container_width = True, type = 'primary'):
                results = get_predictions(USER_IMAGE_NAME)

    if option == RADIO_VIDEO:
        user_video = st.file_uploader("Upload a video", accept_multiple_files=False, help = "Upload any video from your local", type = ["mp4", "avi", "mpeg"])
        if user_video:
            if st.button("Detect Signs", use_container_width = True, type = 'primary'):
                with st.spinner("Processing the Video...."):
                    video_results = get_predictions(USER_VIDEO_NAME)


#create two columns
col1, col2 = st.columns(2)

if user_image:

    with col1:
        st.subheader("User Input")
        #set the user image
        st.image(user_image)

        #read and save the image
        image_bytes = io.BytesIO(user_image.read())
        input_image = Image.open(image_bytes)
        input_image.save(USER_IMAGE_NAME)

if results:
    with col2:
        st.subheader("Prediction")
        #set the user image
        st.image(PREDICTION_IMAGE_PATH)

if user_video:
    #set the user video
    st.subheader("User Input Video")
    st.video(user_video)

    #read bytes
    video_bytes = io.BytesIO(user_video.read())

    with open(USER_VIDEO_NAME, "wb") as video_file:
        video_file.write(video_bytes.read())

    video_file.close()

if video_results:
    st.subheader("Predicted Video")
    st.video(os.path.join(PREDICTION_NEW_PATH, FINAL_PREDICTION_VIDEO))

#if option == RADIO_WEBCAM:
 #   conf = 0.2
    #webrtc_streamer(
      # key="example",
      # mode = WebRtcMode.SENDRECV,
       # video_processor_factory=lambda : model_utils.MyVideoTransformer(conf,model),
       # rtc_configuration={"iceServers": get_ice_servers()},
       # media_stream_constraints={"video": True, "audio": False},
       # async_processing  =True
    #)


#    webrtc_streamer(
 #       key="example",
  #      mode=WebRtcMode.SENDRECV,
   #     rtc_configuration={"iceServers": get_ice_servers()},
    #    media_stream_constraints={"video": True, "audio": False},
     #   )


if option == RADIO_WEBCAM:

    def app_object_detection():
    """Object detection demo with MobileNet SSD."""
    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"
    MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"
    PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class NNVideoTransformer(VideoTransformerBase):
        confidence_threshold: float

        def __init__(self) -> None:
            self._net = cv2.dnn.readNetFromCaffe(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
            self.confidence_threshold = 0.8

        def _annotate_image(self, image, detections):
            # loop over the detections
            (h, w) = image.shape[:2]
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        image,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2,
                    )
            return image

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            image = frame.to_ndarray(format="bgr24")
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self._net.setInput(blob)
            detections = self._net.forward()
            image = self._annotate_image(image, detections)
            return image

    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_class=NNVideoTransformer,
        async_transform=True,
    )


    
