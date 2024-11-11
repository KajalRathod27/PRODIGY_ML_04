import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import json

# Load the trained model
model = load_model('hand_gesture_model.h5')

# Load class labels from the updated JSON file
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Function to process the uploaded image
def process_image(image):
    # Convert the image to an array
    img_array = np.array(image)
    
    # Convert image to RGB if it's not already
    if img_array.shape[-1] != 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Resize image to match the input shape expected by the model (64x64 instead of 128x128)
    img_array = cv2.resize(img_array, (64, 64))  # Resize to 64x64
    
    # Normalize the image
    img_array = img_array / 255.0
    
    # Expand dimensions to add batch size of 1
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Streamlit UI elements
st.title("Hand Gesture Recognition")
st.write("Upload an image of a hand gesture for prediction.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Open the image
    image = Image.open(uploaded_image)

    # Process the image
    processed_img = process_image(image)
    
    # Make prediction
    predictions = model.predict(processed_img)
    
    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    
    # Get the corresponding gesture name from the class labels
    predicted_class_label = [k for k, v in class_labels.items() if v == predicted_class_index][0]
    
    # Display the image and the prediction
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write(f"Prediction: {predicted_class_label}")
    st.write(f"Prediction Probability: {predictions[0][predicted_class_index]*100:.2f}%")
