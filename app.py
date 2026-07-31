
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

STETHOSCOPE_ICON = "https://api.iconify.design/lucide:stethoscope.svg?color=%230284c7"   
WARNING_ICON = "https://api.iconify.design/lucide:triangle-alert.svg?color=%23b45309"
# Page setup
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon=STETHOSCOPE_ICON,
    layout="centered",
)

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["benign tumour", "malignant"] 


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


# UI
#title
st.markdown(
    f'<img src="{STETHOSCOPE_ICON}" width="36" style="vertical-align:middle;margin-right:8px;">'
    f'<span style="font-size:2rem;font-weight:700;">Skin Cancer Classifier</span>',
    unsafe_allow_html=True,
)
st.write(
    "Upload a dermoscopic image of a skin lesion to classify it as "
    "**Skin cancer** or **Benign tumour**."
)
st.markdown(
    f'''
    <div style="background-color:#2d2a1e;border-left:4px solid #b45309;padding:12px 16px;border-radius:4px;display:flex;align-items:flex-start;gap:10px;">
        <img src="{WARNING_ICON}" width="18" style="margin-top:3px;flex-shrink:0;">
        <span style="color:#d4a017;font-size:0.9rem;">
            This tool is a student engineering project for educational purposes only.
            It is <b>not</b> a medical device and must not be used for real diagnosis.
            Always consult a qualified dermatologist.
        </span>
    </div>
    ''',
    unsafe_allow_html=True
)

model = load_model()
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300, caption="Uploaded image")

    with st.spinner("Analysing image..."):
        label, benign_pct, malignant_pct = predict(model, img)

    st.subheader(f"Prediction: **{label.upper()}**")
    st.progress(int(benign_pct), text=f"Benign: {benign_pct:.1f}%")
    st.progress(int(malignant_pct), text=f"Malignant: {malignant_pct:.1f}%")

st.markdown("---")
st.caption(
    "GET 324 — Artificial Intelligence and Machine Learning | "
    "Laboratory Exercise 10 Mini-Project | University of Uyo"
)