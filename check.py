import pandas as pd
import os

df = pd.read_csv("product_dataset.csv", on_bad_lines="skip")

image_folder = "images"

image_ids = {
    os.path.splitext(file)[0]
    for file in os.listdir(image_folder)
    if file.lower().endswith(".jpg")
}

df["id"] = df["id"].astype(str)

matched = df[df["id"].isin(image_ids)]

print("Total CSV products:", len(df))
print("Total images:", len(image_ids))
print("Matching images:", len(matched))

categories = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Watches",
    "Sports Shoes"
]

print("\nImages available for our categories:")

for category in categories:
    count = (matched["articleType"] == category).sum()
    print(category, ":", count)