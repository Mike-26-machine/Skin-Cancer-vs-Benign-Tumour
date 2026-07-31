"""
GET 324 - Laboratory Exercise 10 (Mini-Project)
Streamlit app: Skin Cancer (Malignant) vs Benign Tumour Classifier
Run locally with:  streamlit run app.py
"""



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


# UI
st.title("🩺 Skin Cancer Classifier")
st.write(
    "Upload a dermoscopic image of a skin lesion to classify it as "
    "**Skin cancer** or **Benign tumour**."
)
st.warning(
    "⚠️ This tool is a student engineering project for educational purposes only. "
    "It is **not** a medical device and must not be used for real diagnosis. "
    "Always consult a qualified dermatologist."
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