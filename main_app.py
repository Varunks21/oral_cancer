import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("/gdrive/MyDrive/hybrid_oral_cancer_model.h5")

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to the model's input size
    img_array = img_to_array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit web app
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
