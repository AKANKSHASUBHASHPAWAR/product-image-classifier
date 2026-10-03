import os
import shutil
import pandas as pd

# Load dataset CSV
df = pd.read_csv("product_dataset.csv", on_bad_lines="skip")
df["id"] = df["id"].astype(str)

# 10 Expanded Categories
categories = [
    "Casual Shoes",
    "Formal Shoes",
    "Handbags",
    "Jeans",
    "Kurtas",
    "Shirts",
    "Sports Shoes",
    "Tops",
    "Tshirts",
    "Watches"
]

output_folder = "cnn_dataset"
os.makedirs(output_folder, exist_ok=True)
test_folder = "test_images"
os.makedirs(test_folder, exist_ok=True)

# Create folders for categories
for category in categories:
    os.makedirs(os.path.join(output_folder, category), exist_ok=True)

category_sample_images = {}

print("Copying images into category folders...")
copied_count = 0
for _, row in df.iterrows():
    category = row["articleType"]
    image_id = row["id"]

    if category in categories:
        image_path = os.path.join("images", f"{image_id}.jpg")

        if os.path.exists(image_path):
            destination = os.path.join(output_folder, category, f"{image_id}.jpg")
            if not os.path.exists(destination):
                shutil.copy2(image_path, destination)
            copied_count += 1
            if category not in category_sample_images:
                category_sample_images[category] = image_path

# Populate test_images with one representative image per category
sample_name_map = {
    "Casual Shoes": "casual_shoes.jpg",
    "Formal Shoes": "formal_shoes.jpg",
    "Handbags": "handbag.jpg",
    "Jeans": "jeans.jpg",
    "Kurtas": "kurta.jpg",
    "Shirts": "shirt.jpg",
    "Sports Shoes": "sports_shoes.jpg",
    "Tops": "top.jpg",
    "Tshirts": "tshirt.jpg",
    "Watches": "watch.jpg"
}

for cat, sample_src in category_sample_images.items():
    dest_name = sample_name_map.get(cat, f"{cat.lower().replace(' ', '_')}.jpg")
    dest_path = os.path.join(test_folder, dest_name)
    shutil.copy2(sample_src, dest_path)

# Clean up any malformed files in test_images if present
double_ext_file = os.path.join(test_folder, "shirt.jpg.jpg")
if os.path.exists(double_ext_file):
    try:
        os.remove(double_ext_file)
    except Exception:
        pass

print(f"Dataset preparation completed! Total matching images processed: {copied_count}")
print(f"Sample test images updated in '{test_folder}/'.")