import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

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

# Load class names
if os.path.exists("labels.json"):
    with open("labels.json", "r") as f:
        class_names = json.load(f)
else:
    class_names = validation_data.class_names

# Collect predictions
y_true = []
y_pred = []

for images, labels in validation_data:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix - 10-Class Product Image Classifier", fontsize=14, pad=15)
plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
plt.ylabel("Actual Label", fontsize=12, labelpad=10)

plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=10)
plt.yticks(range(len(class_names)), class_names, fontsize=10)

# Add numbers inside cells
thresh = cm.max() / 2.0
for i in range(len(class_names)):
    for j in range(len(class_names)):
        color = "white" if cm[i, j] > thresh else "black"
        plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color, fontsize=10)

plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()

# Save image
output_file = "confusion_matrix.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Confusion matrix saved successfully to: {output_file}")