import os

dataset_folder = "cnn_dataset"

if not os.path.exists(dataset_folder):
    print(f"Error: '{dataset_folder}' does not exist.")
    exit(1)

categories = sorted([
    d for d in os.listdir(dataset_folder)
    if os.path.isdir(os.path.join(dataset_folder, d))
])

print("========================================")
print(" Product Dataset Verification Summary   ")
print("========================================")

total_images = 0
category_counts = {}

for category in categories:
    folder = os.path.join(dataset_folder, category)
    images = [
        file for file in os.listdir(folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    count = len(images)
    category_counts[category] = count
    total_images += count

for category, count in category_counts.items():
    pct = (count / total_images * 100) if total_images > 0 else 0
    print(f"{category:15} : {count:4d} images ({pct:5.1f}%)")

print("----------------------------------------")
print(f"Total Categories: {len(categories)}")
print(f"Total Images:     {total_images}")
print("========================================")