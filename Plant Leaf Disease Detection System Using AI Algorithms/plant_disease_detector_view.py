import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Title
st.title("🌿 Plant Leaf Disease Detection")
st.subheader("Upload a leaf image to detect the disease")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fast_test_model.h5")

model = load_model()

# Class labels (must match your training classes!)
class_labels = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot',
    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Uploaded Leaf Image', use_container_width=True)

    # Preprocess
    img = image_data.resize((128, 128))  # Match training size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_labels[np.argmax(predictions)]

    # Show result
    st.success(f"✅ Predicted Disease: **{predicted_class}**")

