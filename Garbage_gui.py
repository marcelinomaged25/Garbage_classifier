import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

# Load model once
@st.cache_resource
def load_model():
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])
    
    model.load_weights('model_weights_22.weights.h5')
    return model

model = load_model()

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)  
    return image


st.title("♻️ Waste Classification")
st.write("Upload an image to classify it into: **cardboard**, **glass**, **metal**, **paper**, **plastic**, or **trash**.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.subheader("🔍 Prediction Results")
        
        st.write(f"**Predicted Class:** {predicted_class}")
        if predicted_class in {'cardboard', 'glass', 'metal', 'paper', 'plastic'}:
            st.write(f"**Recyclable:** ✅ Yes")
        else:
            st.write(f"**Recyclable:** ❌ No")
        st.write(f"**Confidence:** {confidence:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error processing the image: {e}")
        
