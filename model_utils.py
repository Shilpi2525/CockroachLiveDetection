from moviepy.editor import VideoFileClip
from ultralytics import YOLO
import streamlit as st
import numpy as np
from PIL import Image

def convert_mp4_H264(input_file: str, output_file: str) -> None:
    # Load the video clip
    clip = VideoFileClip(input_file)

    # Set the codec to H.264
    codec = "libx264"

    # Save the video with the specified codec
    clip.write_videofile(output_file, codec=codec)

    print("Conversion complete.")

@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)
