import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load model
model = tf.keras.models.load_model("product_classifier.keras")

# Load validation data
validation_data = tf.keras.utils.image_dataset_from_directory(
    "cnn_dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

class_names = validation_data.class_names

# Get actual and predicted labels
y_true = []
y_pred = []

for images, labels in validation_data:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(xticks_rotation=45)

plt.title("Product Image Classifier - Confusion Matrix")
plt.tight_layout()
plt.show()