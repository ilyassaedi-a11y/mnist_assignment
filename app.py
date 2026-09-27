import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")


model = load_model()


st.title("🔢 MNIST Digit Recognition")
st.write("Upload a handwritten digit image and let the AI predict it.")


uploaded_file = st.file_uploader(
    "Upload a handwritten digit",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    if st.button("Predict Digit", type="primary"):

        # Convert image to grayscale
        image = image.convert("L")

        # Resize to MNIST size
        image = image.resize((28, 28))

        # Convert to numpy array
        image_array = np.array(image)

        # Invert if necessary
        if image_array.mean() > 127:
            image_array = 255 - image_array

        # Normalize
        image_array = image_array.astype("float32") / 255.0

        # Add batch and channel dimensions
        image_array = image_array.reshape(1, 28, 28, 1)

        # Prediction
        prediction = model.predict(image_array)

        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        st.success("Prediction Complete!")

        st.metric(
            "Predicted Digit",
            digit
        )

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )