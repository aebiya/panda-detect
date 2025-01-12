import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load a pre-trained MobileNetV2 model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

model = load_model()

# Function to process and predict
def predict(image):
    image = image.resize((224, 224))  # Resize for MobileNetV2
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    return decoded_predictions

# Streamlit App
st.title("Panda Detection App")
st.write("Upload an image, and the app will detect if there's a panda in it!")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Detecting...")
    results = predict(image)

    for i, res in enumerate(results[0]):
        label, description, score = res
        st.write(f"{i+1}. **{description}** - Confidence: {score:.2%}")

    # Highlight if Panda detected
    panda_result = next((res for res in results[0] if "panda" in res[1].lower()), None)
    if panda_result:
        st.success(f"Panda detected with confidence: {panda_result[2]:.2%}")
    else:
        st.error("No panda detected.")

