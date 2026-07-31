

# model = load_model()
# uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     img = Image.open(uploaded_file)
#     st.image(img, width=300, caption="Uploaded image")

#     with st.spinner("Analysing image..."):
#         label, benign_pct, malignant_pct = predict(model, img)

#     st.subheader(f"Prediction: **{label.upper()}**")
#     st.progress(int(benign_pct), text=f"Benign: {benign_pct:.1f}%")
#     st.progress(int(malignant_pct), text=f"Malignant: {malignant_pct:.1f}%")

# st.markdown("---")
# st.caption(
#     "GET 324 — Artificial Intelligence and Machine Learning | "
#     "Laboratory Exercise 10 Mini-Project | University of Uyo"
# )