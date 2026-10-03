import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

model_path = "best_product_classifier.keras"
if not os.path.exists(model_path):
    model_path = "product_classifier.keras"

print(f"Loading model: {model_path}")
model = tf.keras.models.load_model(model_path)

# Load validation dataset
validation_data = tf.keras.utils.image_dataset_from_directory(
    "cnn_dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

# Load class names from labels.json if available
if os.path.exists("labels.json"):
    with open("labels.json", "r") as f:
        class_names = json.load(f)
else:
    class_names = validation_data.class_names

print(f"Evaluating {len(class_names)} categories...")

y_true = []
y_pred = []

for images, labels in validation_data:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

print("\n=======================================================")
print("             DETAILED CLASSIFICATION REPORT            ")
print("=======================================================")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

print("\n=======================================================")
print("                   CONFUSION MATRIX                    ")
print("=======================================================")
cm = confusion_matrix(y_true, y_pred)
print(cm)