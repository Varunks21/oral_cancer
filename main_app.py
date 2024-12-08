import os
import gdown  # Install with `pip install gdown`
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np

# Define model path and Google Drive file ID
model_path = "hybrid_oral_cancer_model.h5"
file_id = "1SjrH-KLR-fZHd9lhKkODnykbDZqs1emK"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

# Download model if not already present
if not os.path.exists(model_path):
    st.write("Downloading model...")
    gdown.download(gdrive_url, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)
st.success("Model loaded successfully!")

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to the model's input size
    img_array = img_to_array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app interface
st.title("Oral Cancer Detection Web App")
st.write("Upload an image to classify as **Cancer** or **Non-Cancer**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image and make prediction
    st.write("Processing...")
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    
    # Display the result
    result = "Cancer" if prediction[0] > 0.5 else "Non-Cancer"
    confidence = prediction[0][0] if prediction[0] < 0.5 else 1 - prediction[0][0]
    st.write(f"Prediction: **{result}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
