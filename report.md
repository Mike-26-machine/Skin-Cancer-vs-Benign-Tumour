# Project Report (Draft — 100–150 words)

Our group developed a binary image classifier distinguishing malignant from benign
skin lesions, using the "Skin Cancer: Malignant vs Benign" dataset from the ISIC
Archive (Kaggle, fanconic). We trained a custom CNN and a MobileNetV3 transfer-learning
model in TensorFlow/Keras, then deployed the better-performing model as a Streamlit web
application that lets a user upload a dermoscopic image and receive a benign/malignant
prediction with confidence scores.



Future improvements could include collecting a larger, more balanced dataset and adding
Grad-CAM visual explanations to build clinical trust in predictions.