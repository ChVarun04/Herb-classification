import streamlit as st
import numpy as np
from PIL import Image

from utils.predictor import predict_leaf
from utils.webcam import start_webcam


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Medicinal Leaf Recognition",
    page_icon="🌿",
    layout="wide"
)


# TITLE
st.title("🌿 Medicinal Leaf Recognition")

st.write(
    "Medicinal Leaf Recognition using "
    "MediaPipe MobileNetV3 + Artificial Neural Network"
)


# SIDEBAR
st.sidebar.title("Input Mode")

mode = st.sidebar.radio(
    "Choose Input",
    [
        "📁 Upload Image",
        "📷 Live Webcam"
    ]
)


# IMAGE UPLOAD
if mode == "📁 Upload Image":

    st.header("📁 Upload Leaf Image")

    uploaded = st.file_uploader(
        "Choose a medicinal leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:

        image = Image.open(
            uploaded
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Leaf",
            width=400
        )

        image_np = np.array(image)

        leaf_name, confidence = predict_leaf(
            image_np
        )

        if leaf_name is None:

            st.error(
                "Unable to recognize the leaf."
            )

        else:

            st.success(
                f"🌿 Prediction: {leaf_name}"
            )

            st.info(
                f"🎯 Confidence: "
                f"{confidence * 100:.2f}%"
            )


# LIVE WEBCAM
else:

    st.header("📷 Live Webcam Detection")

    st.write(
        "Show a medicinal leaf in front "
        "of your webcam."
    )

    start_webcam()