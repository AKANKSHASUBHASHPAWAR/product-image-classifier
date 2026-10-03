import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Product Image Classifier",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------
# Load trained model & class labels
# -----------------------------
@st.cache_resource
def load_classifier():
    model_file = "product_classifier.keras"
    model = tf.keras.models.load_model(model_file)
    labels_file = "labels.json"
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            classes = json.load(f)
    else:
        classes = [
            "Casual Shoes", "Formal Shoes", "Handbags", "Jeans",
            "Kurtas", "Shirts", "Sports Shoes", "Tops", "Tshirts", "Watches"
        ]
    return model, classes

model, class_names = load_classifier()

# -----------------------------
# Sidebar: Information & Settings
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shopping-cart-loaded.png", width=64)
    st.title("Product Classifier")
    st.markdown("Deep Learning classifier trained on e-commerce catalog images.")

    st.markdown("---")
    st.subheader(f"Supported Categories ({len(class_names)})")
    for idx, name in enumerate(class_names, 1):
        st.caption(f"**{idx}.** {name}")

    st.markdown("---")
    confidence_threshold = st.slider(
        "Confidence Alert Threshold (%)",
        min_value=30,
        max_value=90,
        value=50,
        step=5,
        help="Alerts if top prediction falls below this threshold."
    )
    st.info("Tip: Use high-contrast photos against plain backgrounds for best results.")

# -----------------------------
# Main Header
# -----------------------------
st.title("🛍️ Smart Product Image Classifier")
st.markdown("Upload or choose a product image to identify its category and inspect prediction probabilities.")

# -----------------------------
# Input Selection: Samples vs Upload
# -----------------------------
input_tab1, input_tab2, input_tab3 = st.tabs(["📁 Sample Gallery", "📤 Upload Image", "📷 Live Camera"])

selected_image = None
image_source = None

# Tab 1: Sample Gallery
with input_tab1:
    st.write("Click any sample image below to instantly test the model:")
    test_folder = "test_images"
    if os.path.exists(test_folder):
        sample_files = [f for f in sorted(os.listdir(test_folder)) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        cols = st.columns(min(len(sample_files), 5))
        for i, file_name in enumerate(sample_files):
            col = cols[i % len(cols)]
            clean_title = file_name.replace(".jpg", "").replace(".png", "").replace("_", " ").title()
            img_path = os.path.join(test_folder, file_name)
            with col:
                st.image(img_path, caption=clean_title, use_container_width=True)
                if st.button("Classify", key=f"btn_{file_name}"):
                    selected_image = Image.open(img_path).convert("RGB")
                    image_source = file_name

# Tab 2: Upload Image
with input_tab2:
    uploaded_file = st.file_uploader(
        "Choose an image from your device",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file).convert("RGB")
        image_source = uploaded_file.name

# Tab 3: Camera
with input_tab3:
    camera_file = st.camera_input("Take a photo of a product")
    if camera_file is not None:
        selected_image = Image.open(camera_file).convert("RGB")
        image_source = "Camera Capture"

# -----------------------------
# Prediction & Results Display
# -----------------------------
if selected_image is not None:
    st.markdown("---")
    col_img, col_pred = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Selected Image")
        st.image(selected_image, caption=f"Source: {image_source}", use_container_width=True)

    with col_pred:
        st.subheader("Prediction Results")

        # Preprocessing
        resized_image = selected_image.resize((128, 128))
        image_array = np.array(resized_image)
        image_array = np.expand_dims(image_array, axis=0)

        # Inference
        predictions = model.predict(image_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1]

        best_index = top_indices[0]
        best_class = class_names[best_index]
        best_confidence = predictions[best_index] * 100

        # Display Top-1 Metric
        st.success(f"### Predicted: **{best_class}**")
        st.metric(label="Top Prediction Confidence", value=f"{best_confidence:.2f}%")

        if best_confidence < confidence_threshold:
            st.warning(f"⚠️ Confidence is below {confidence_threshold}%. The product might be ambiguous or out of catalog.")

        # Top-3 Probabilities
        st.markdown("#### Top-3 Category Candidates")
        for rank in range(min(3, len(class_names))):
            idx = top_indices[rank]
            cat_name = class_names[idx]
            conf = predictions[idx] * 100
            st.write(f"**{rank + 1}. {cat_name}** — `{conf:.2f}%`")
            st.progress(float(predictions[idx]))

        # Expandable all classes
        with st.expander("View Full Probability Distribution"):
            prob_dict = {class_names[i]: float(predictions[i] * 100) for i in top_indices}
            st.bar_chart(prob_dict)
