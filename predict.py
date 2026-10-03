import os
import json
import numpy as np
import tensorflow as tf

model_path = "best_product_classifier.keras"
if not os.path.exists(model_path):
    model_path = "product_classifier.keras"

print(f"Loading model: {model_path}")
model = tf.keras.models.load_model(model_path)

# Load class names dynamically
labels_path = "labels.json"
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        class_names = json.load(f)
else:
    class_names = [
        "Casual Shoes", "Formal Shoes", "Handbags", "Jeans",
        "Kurtas", "Shirts", "Sports Shoes", "Tops", "Tshirts", "Watches"
    ]

test_folder = "test_images"
if not os.path.exists(test_folder):
    print(f"Error: Folder '{test_folder}' does not exist.")
    exit(1)

test_files = [
    f for f in sorted(os.listdir(test_folder))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"\nEvaluating {len(test_files)} sample test images across {len(class_names)} categories:\n")

for image_name in test_files:
    image_path = os.path.join(test_folder, image_name)

    image = tf.keras.utils.load_img(image_path, target_size=(128, 128))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:3]

    best_idx = top_indices[0]
    best_class = class_names[best_idx]
    best_conf = predictions[best_idx] * 100

    print("-------------------------------------------------------")
    print(f"File:               {image_name}")
    print(f"Top-1 Prediction:   {best_class} ({best_conf:.2f}%)")
    print("Top-3 Breakdown:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  #{rank} {class_names[idx]:15} : {predictions[idx] * 100:6.2f}%")
print("=======================================================")