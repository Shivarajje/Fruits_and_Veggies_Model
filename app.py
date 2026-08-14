import streamlit as st
import numpy as np
import tf_keras as keras
from PIL import Image, ImageOps

# Page Configuration
st.set_page_config(
    page_title="Fruit & Vegetable Identifier",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 Fruit & Vegetable Identifier")
st.write("Upload an image to identify the item and view classification confidence.")

# 1. Load the Model with Caching
@st.cache_resource
def load_classification_model():
    model = keras.models.load_model('model.h5', compile=False)
    return model

# 2. Load the Class Labels
@st.cache_resource
def load_labels():
    class_labels = {}
    with open('labels.txt', 'r') as f:
        for line in f:
            if line.strip():
                idx, label = line.strip().split(' ', 1)
                class_labels[int(idx)] = label
    return class_labels

model = load_classification_model()
class_labels = load_labels()

# 3. File Uploader Interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        # Preprocessing (224x224, [-1, 1] normalization)
        img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(img_resized, dtype=np.float32)
        normalized_img_array = (img_array / 127.5) - 1.0
        data = np.expand_dims(normalized_img_array, axis=0)

        # Predict
        predictions = model.predict(data)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100
        predicted_label = class_labels.get(predicted_idx, "Unknown")

    # Display Results
    st.success(f"**Prediction:** {predicted_label}")
    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
    st.progress(int(confidence))