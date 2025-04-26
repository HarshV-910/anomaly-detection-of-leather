import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps  # Pillow library

# Load labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Function to predict with TFLite model
def predict_image_tflite(image_file):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load image
    image = Image.open(image_file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    
    # Normalize image to [-1, 1]
    normalized_image_array = (image_array / 127.5) - 1.0

    # Add batch dimension
    input_data = np.expand_dims(normalized_image_array, axis=0)

    # Set tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output_data)

    # Get highest confidence index
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[index]

    return class_name, confidence_score

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Data", "Leather Anomaly Detection"])

# Main Page
if app_mode == "Home":
    st.header("Marble Surface Anomaly Detection")
    st.header(tf.__version__)
    st.image("https://imgs.search.brave.com/eBvLVlfi8-X1GfzPxRWQ84WwsUpwsCkRFfYxKYTAZOg/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4u/c2hvcGlmeS5jb20v/cy9maWxlcy8xLzAw/MjkvMDM5Ni81NzYw/L2ZpbGVzLzFfMTEx/MTFmYTEtOWY1OC00/OGNhLTliNDYtM2U2/MDkyYzJmNzVjXzEw/MjR4MTAyNC5wbmc_/dj0xNjIyMjE3MzY3")
    st.markdown("""
        Awesome — you’re tackling a meaningful and highly practical computer vision problem! Let’s lay out a clean overview for your **Leather Anomaly Detection in Manufacturing** project. You can use this in your documentation, Streamlit app, or report:

---

## 📌 Leather Anomaly Detection in Manufacturing

### 📖 Project Overview:
In leather manufacturing industries, ensuring the quality of leather products is critical. Defects like color inconsistencies, physical damages, or improper processing can affect product quality and market value. This project automates the **visual inspection process** by using **computer vision and deep learning** techniques to detect different types of surface anomalies in leather products.

---

### 🎯 Objective:
To build an automated system that can classify leather images into various categories based on surface conditions, improving inspection efficiency and reducing manual errors.

---


### 🔍 Importance:
- Reduces dependency on manual inspection.
- Minimizes faulty deliveries.
- Ensures consistent product quality.
- Helps in early detection of process errors.
- Speeds up the quality control workflow.

        """)
    
elif app_mode == "About Data":
    st.header("About Data")
    st.markdown("""
        Excellent — let’s flesh out a proper **Dataset Description** section for your Leather Anomaly Detection project. You can use this in your report, documentation, or Streamlit app as well.

---

## 📚 Dataset Description: Leather Anomaly Detection

### 📖 Context:
In leather manufacturing industries, visual inspection is a critical quality control step. Manual inspection, however, can be **inconsistent, time-consuming, and prone to human error**. To address this, we curated a dataset of high-resolution leather surface images, categorized into multiple defect classes typically encountered during the manufacturing process.

---

### 📊 Dataset Structure:
The dataset is organized into **6 distinct classes** representing different types of surface conditions:

| Class   | Description                                     | Example Issues                        |
|:--------|:------------------------------------------------|:--------------------------------------|
| **Color**  | Irregular color patches, discoloration         | Faded spots, uneven tanning            |
| **Cut**    | Visible cuts, tears, or slits on the surface    | Knife cuts, accidental slashes        |
| **Poke**   | Small holes or puncture defects                | Needle holes, sharp object pokes      |
| **Good**   | Defect-free, high-quality leather surfaces     | Smooth, consistent texture and color  |
| **Fold**   | Wrinkles or folds in the leather               | Creases from handling or storage      |
| **Glue**   | Visible glue stains or adhesive marks          | Processing adhesive residues          |

---

### 🖼️ Image Details:
- **Image Dimensions**: Variable (e.g., 128x128, 224x224) depending on preprocessing.
- **Image Type**: RGB images captured using high-resolution cameras in industrial setups.
- **Format**: JPEG / PNG.

---

### 📂 Dataset Directory Structure:
```
/leather_anomalies_dataset/
 ├── color/
 │    ├── img1.jpg
 │    ├── img2.jpg
 │    └── ...
 ├── cut/
 ├── poke/
 ├── good/
 ├── fold/
 └── glue/
```

---

### 📑 Dataset Purpose:
- Train, validate, and test deep learning models for image classification.
- Develop automated anomaly detection systems for leather manufacturing.
- Benchmark different CNN architectures and preprocessing techniques.

---

### 📊 Possible Future Extensions:
- **Add more defect classes**: (e.g., stain, scratch, burn mark)
- **Pixel-wise defect segmentation** for localization.
- **Anomaly detection using autoencoders** for unseen defect types.

---

Would you like a sample `.csv` label file structure too (if you’re managing labels via a CSV for your pipeline)? I can draft that too if needed!

        """)

elif app_mode == "Leather Anomaly Detection":
    st.header("Leather Anomaly Detection")
    test_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_container_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Prediction in progress...")

            class_name, confidence_score = predict_image_tflite(test_image)

            st.success(f"**Prediction:** {class_name}")
            st.info(f"**Confidence Score:** {confidence_score:.2f}")
