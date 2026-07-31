"""
GET 324 - Laboratory Exercise 10 (Mini-Project)
Streamlit app: Skin Cancer (Malignant) vs Benign Tumour Classifier
Run locally with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


# Page setup
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🩺",
    layout="centered",
)

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["benign tumour", "malignant"]  # must match training folder order


# Load the trained model once and cache it across reruns
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/mobilenetv3_transfer.keras")
    return model


def predict(model, pil_image):
    """Preprocess the uploaded image and return the predicted label + confidence."""
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    prob_malignant = float(model.predict(arr, verbose=0)[0][0])
    prob_benign = 1.0 - prob_malignant
    label = CLASS_NAMES[1] if prob_malignant >= 0.5 else CLASS_NAMES[0]
    return label, prob_benign * 100, prob_malignant * 100

