import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

dataset_path = "cnn_dataset"

# -----------------------------
# LOAD DATASETS
# -----------------------------
train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

class_names = train_data.class_names
num_classes = len(class_names)

print(f"\nDiscovered {num_classes} classes: {class_names}")

# Save class labels dynamically for inference and Streamlit app
labels_path = "labels.json"
with open(labels_path, "w") as f:
    json.dump(class_names, f, indent=4)
print(f"Saved class names to {labels_path}")

# -----------------------------
# COMPUTE BALANCED CLASS WEIGHTS
# -----------------------------
print("\nComputing balanced class weights...")
y_train_list = []
for _, labels in train_data:
    y_train_list.extend(labels.numpy())

y_train_arr = np.array(y_train_list)
unique_classes = np.unique(y_train_arr)
computed_weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train_arr
)
# Moderate the weights with square root to prevent over-penalizing majority classes
moderated_weights = np.sqrt(computed_weights)
class_weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, moderated_weights)}
for cls_idx, w in class_weight_dict.items():
    print(f"  {class_names[cls_idx]:15} (class {cls_idx}): weight = {w:.2f}")

# -----------------------------
# CACHING & PREFETCHING
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = validation_data.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05)
], name="data_augmentation")

# -----------------------------
# BUILD MOBILENETV2 TRANSFER LEARNING MODEL
# -----------------------------
print("\nBuilding MobileNetV2 Transfer Learning Model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze pretrained weights for high stability and fast CPU training

inputs = layers.Input(shape=(128, 128, 3), name="input_image")
# Preprocess images to [-1, 1] as required by MobileNetV2
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = data_augmentation(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu", name="fc_dense")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_product_classifier")
model.summary()

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# CALLBACKS
# -----------------------------
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
    verbose=1
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_product_classifier.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
epochs = 20
print(f"\nStarting training for up to {epochs} epochs...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weight_dict,
    callbacks=[
        early_stopping,
        reduce_lr,
        model_checkpoint
    ]
)

# -----------------------------
# SAVE FINAL MODEL & HISTORY
# -----------------------------
model.save("product_classifier.keras")

# Clean history into standard Python floats for JSON serialization
clean_history = {k: [float(val) for val in v] for k, v in history.history.items()}
with open("training_history.json", "w") as f:
    json.dump(clean_history, f, indent=4)

print("\n==============================================")
print("Training successfully completed!")
print("Best model saved as:     best_product_classifier.keras")
print("Final model saved as:    product_classifier.keras")
print("Class labels saved as:   labels.json")
print("Training history saved:  training_history.json")
print("==============================================")