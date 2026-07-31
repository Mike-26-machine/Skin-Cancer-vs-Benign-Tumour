"""
GET 324 - Laboratory Exercise 10 (Mini-Project)
Binary Image Classification: Skin Cancer (Malignant) vs Benign Tumours

Dataset: "Skin Cancer: Malignant vs Benign" (Kaggle, fanconic)
          https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

This script trains a custom CNN AND a MobileNetV3 transfer-learning model,
compares them, and saves the better one to models/ for the Streamlit app.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


# STEP 1: Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

results_dir = "results/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs("models", exist_ok=True)

# STEP 2: Dataset paths — the Kaggle dataset only ships train/ and test/,
# so we carve a validation split out of train/ using validation_split.
DATA_DIR = Path("C:\Users\OMEN\OneDrive\Documents\GET324_AI_ML\mini-project 2\data")
train_dir = DATA_DIR / "train"
test_dir = DATA_DIR / "test"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE,
    validation_split=0.2, subset="training", seed=SEED, label_mode="binary",
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE,
    validation_split=0.2, subset="validation", seed=SEED, label_mode="binary",
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE,
    shuffle=False, label_mode="binary",
)

class_names = train_dataset.class_names  # e.g. ['benign', 'malignant']
print(f"Classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(AUTOTUNE)


# STEP 3: Data augmentation (train only)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
], name='data_augmentation')

# STEP 4: Shared callbacks + evaluation helpers
def make_callbacks(name):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"models/{name}_best.keras",
            monitor="val_accuracy", save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1,
        ),
    ]


def plot_learning_curves(history, title):
    acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
    loss, val_loss = history.history["loss"], history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, "--", label="Val Accuracy")
    plt.legend(); plt.title(f"{title} — Accuracy"); plt.xlabel("Epoch"); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, "--", label="Val Loss")
    plt.legend(); plt.title(f"{title} — Loss"); plt.xlabel("Epoch"); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_dir}{title.lower().replace(' ', '_')}_curves.png", dpi=120)
    plt.show()


def evaluate_model(model, dataset, title):
    y_true, y_pred = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).ravel()
        y_pred.append((probs >= 0.5).astype(int))
        y_true.append(labels.numpy().ravel().astype(int))
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    print(f"\n{title}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{title} — Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{results_dir}{title.lower().replace(' ', '_')}_confusion.png", dpi=120)
    plt.show()

    return accuracy_score(y_true, y_pred)



# STEP 5: Custom CNN (binary — sigmoid output)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

def build_custom_cnn(input_shape, augmentation):
    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # binary
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


cnn_model = build_custom_cnn(INPUT_SHAPE, data_augmentation)
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy", metrics=["accuracy"],
)
cnn_history = cnn_model.fit(
    train_dataset, validation_data=val_dataset, epochs=EPOCHS,
    callbacks=make_callbacks("custom_cnn"),
)
plot_learning_curves(cnn_history, "Custom CNN")
cnn_test_acc = evaluate_model(cnn_model, test_dataset, "Custom CNN")


# STEP 6: Transfer learning — MobileNetV3Small (binary — sigmoid output)
def build_transfer_model(input_shape, augmentation):
    base_model = tf.keras.applications.MobileNetV3Small(
        weights="imagenet", input_shape=input_shape, include_top=False,
    )
    base_model.trainable = False
    preprocess_fn = tf.keras.applications.mobilenet_v3.preprocess_input

    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = preprocess_fn(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv3_transfer"), base_model


tl_model, base_model = build_transfer_model(INPUT_SHAPE, data_augmentation)
tl_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy", metrics=["accuracy"],
)
tl_history = tl_model.fit(
    train_dataset, validation_data=val_dataset, epochs=EPOCHS,
    callbacks=make_callbacks("tl_feature_extraction"),
)
plot_learning_curves(tl_history, "MobileNetV3 Transfer Learning")
tl_test_acc = evaluate_model(tl_model, test_dataset, "MobileNetV3 Transfer")

# Optional fine-tuning of the top layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

tl_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy", metrics=["accuracy"],
)
ft_history = tl_model.fit(
    train_dataset, validation_data=val_dataset, epochs=15,
    callbacks=make_callbacks("tl_finetuned"),
)
plot_learning_curves(ft_history, "MobileNetV3 Fine-Tuned")
ft_test_acc = evaluate_model(tl_model, test_dataset, "MobileNetV3 Fine-Tuned")

# STEP 7: Compare and save both models (app.py picks whichever performed best)
results = {
    "Custom CNN": cnn_test_acc,
    "MobileNetV3 (Feature Extraction)": tl_test_acc,
    "MobileNetV3 (Fine-Tuned)": ft_test_acc,
}
print(f"\n{'Model':<35} {'Test Accuracy':>15}")
print("-" * 52)
for name, acc in results.items():
    print(f"{name:<35} {acc:>15.4f}")

cnn_model.save("models/custom_cnn.keras")
tl_model.save("models/mobilenetv3_transfer.keras")
print("\nModels saved to models/custom_cnn.keras and models/mobilenetv3_transfer.keras")
