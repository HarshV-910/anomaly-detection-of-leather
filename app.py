# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from keras.models import load_model  # TensorFlow is required for Keras to work
# from PIL import Image, ImageOps  # Install pillow instead of PIL

# #Tensorflow Model Prediction
# def model_prediction(test_image):
#     model = tf.keras.models.load_model("trained_plant_disease_model.keras")
#     image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr]) #convert single image to batch
#     predictions = model.predict(input_arr)
#     return np.argmax(predictions) #return index of max element


# def predict_image(test_image):
#     # Disable scientific notation for clarity
#     np.set_printoptions(suppress=True)

#     # Load the model
#     model = load_model("keras_model.h5", compile=False)

#     # Load the labels
#     class_names = open("labels.txt", "r").readlines()

#     # Create the array of the right shape to feed into the keras model
#     # The 'length' or number of images you can put into the array is
#     # determined by the first position in the shape tuple, in this case 1
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#     # Replace this with the path to your image
#     image = Image.open("test_image").convert("RGB")

#     # resizing the image to be at least 224x224 and then cropping from the center
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

#     # turn the image into a numpy array
#     image_array = np.asarray(image)

#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

#     # Load the image into the array
#     data[0] = normalized_image_array

#     # Predicts the model
#     prediction = model.predict(data)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]

#     # Print prediction and confidence score
#     print("Class:", class_name[2:], end="")
#     print("Confidence Score:", confidence_score)




# #Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# #Main Page
# if(app_mode=="Home"):
#     st.header("Marble Surface Anomaly Detection")
#     st.header(tf.__version__)
#     image_path = "https://imgs.search.brave.com/plgifbgjzYCmb74OLcLVoIHMOewTDvmN448BuZM4qtw/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy85/LzkyL01hcmJsZV93/YWxsX29mX1J1c2tl/YWxhLmpwZw"
#     st.image(image_path)
#     st.markdown("""

# ## 🏛️ What is Marble Surface Anomaly Detection?

# It’s a **computer vision-based inspection task** where the goal is to automatically detect defects or anomalies on marble surfaces (tiles, slabs, or other products). These defects can include:
# - **Cracks**
# - **Stains**
# - **Color inconsistency**
# - **Spots**
# - **Holes**
# - **Vein pattern irregularities**
# - **Surface roughness**

# Since marble is a **natural material**, every slab has a unique pattern — which makes anomaly detection a challenging task because you need to differentiate between natural variations and actual defects.

# ---

# ## 📦 Types of Defects Typically Detected:
# | Type         | Description                        |
# |:------------|:-----------------------------------|
# | Cracks       | Thin lines or breaks               |
# | Holes        | Small missing parts in surface     |
# | Stains       | Discolorations of varying shades   |
# | Dark Spots   | Unusual darker patches             |
# | Pattern Breaks | Discontinuity in expected patterns |

# ---

# ## 🖥️ How is it Done?

# ### 1️⃣ Traditional Computer Vision:
# - **Image acquisition** (camera or scanner)
# - **Preprocessing**: filtering, denoising, contrast enhancement
# - **Edge detection** (Canny, Sobel)
# - **Texture analysis**: GLCM (Gray Level Co-occurrence Matrix), LBP (Local Binary Patterns)
# - **Thresholding and segmentation**
# - **Contour detection** to isolate defective regions

# ---

# ### 2️⃣ Machine Learning-based:
# - Extract handcrafted features (texture, color histograms, shape descriptors)
# - Train classifiers like:
#   - SVM
#   - Random Forest
#   - k-NN
# - On labeled defective/non-defective patches

# ---

# ### 3️⃣ Deep Learning-based (State-of-the-Art 🔥)
# - Use **Convolutional Neural Networks (CNNs)** for automated feature extraction
# - Techniques:
#   - **Image classification CNNs**: classify entire tile as defective / non-defective
#   - **Object detection CNNs** (YOLO, Faster R-CNN): locate defect regions
#   - **Anomaly detection autoencoders**: detect abnormal regions by reconstruction error
#   - **Segmentation CNNs** (like U-Net): pixel-wise detection of defects


#     """)

# #About Project
# elif(app_mode=="About"):
#     st.header("About")
#     st.markdown("""
#                 #### About Dataset
#                 Context
#                 Marble Surface Anomaly Detection dataset is a raw dataset that the user can mould in any form that deems fit to the work. The images are taken as they will be in the real industrial setup.
#                 A smartphone camera is used to capture the images.
#                 I will soon put a more comprehensive dataset that can be directly consumed by the models

#                 Content
#                 The images are **2218 x 4608** in dimension. Currently, there are only two folders good and defect.

#                 """)

# #Prediction Page
# elif(app_mode=="Disease Recognition"):
#     st.header("Disease Recognition")
#     test_image = st.file_uploader("Choose an Image:")
#     if(st.button("Show Image")):
#         st.image(test_image)
#     #Predict button
#     if(st.button("Predict")):
#         st.snow()
#         st.write("Our Prediction")
#         # result_index = model_prediction(test_image)
#         predict_image(test_image)


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
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("Marble Surface Anomaly Detection")
    st.header(tf.__version__)
    st.image("https://imgs.search.brave.com/plgifbgjzYCmb74OLcLVoIHMOewTDvmN448BuZM4qtw/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy85/LzkyL01hcmJsZV93/YWxsX29mX1J1c2tl/YWxhLmpwZw")
    st.markdown("""
        ## 🏛️ What is Marble Surface Anomaly Detection?
        It’s a **computer vision-based inspection task** for detecting defects or anomalies on marble surfaces.
        """)
    
elif app_mode == "About":
    st.header("About")
    st.markdown("""
        #### About Dataset
        Marble Surface Anomaly Detection dataset mimics real industrial setups, captured via smartphone camera.
        Images are **2218 x 4608** and classified into `good` and `defect`.
        """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Prediction in progress...")

            class_name, confidence_score = predict_image_tflite(test_image)

            st.success(f"**Prediction:** {class_name}")
            st.info(f"**Confidence Score:** {confidence_score:.2f}")
