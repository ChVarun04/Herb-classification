import cv2
import numpy as np
import joblib
import mediapipe as mp
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# LOAD MODELS
ANN_MODEL = tf.keras.models.load_model(
    "models/medicinal_leaf_ann.keras"
)

SCALER = joblib.load(
    "models/scaler.pkl"
)

LABEL_ENCODER = joblib.load(
    "models/label_encoder.pkl"
)


# MEDIAPIPE IMAGE EMBEDDER
BaseOptions = python.BaseOptions

options = vision.ImageEmbedderOptions(
    base_options=BaseOptions(
        model_asset_path=
        "models/mobilenet_v3_small.tflite"
    ),
    l2_normalize=True,
    quantize=False
)

embedder = vision.ImageEmbedder.create_from_options(
    options
)


# FEATURE EXTRACTION
def extract_features(image):

    if image.shape[-1] == 4:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2RGB
        )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )

    result = embedder.embed(
        mp_image
    )

    if len(result.embeddings) == 0:
        return None

    emb = result.embeddings[0]

    if hasattr(emb, "float_embedding"):

        feature = np.array(
            emb.float_embedding
        )

    elif hasattr(emb, "embedding"):

        feature = np.array(
            emb.embedding
        )

    elif hasattr(emb, "numpy_view"):

        feature = emb.numpy_view()

    else:

        return None

    feature = feature.reshape(
        1,
        -1
    )

    feature = SCALER.transform(
        feature
    )

    return feature


# PREDICTION
def predict_leaf(image):

    feature = extract_features(
        image
    )

    if feature is None:
        return None, 0

    prediction = ANN_MODEL.predict(
        feature,
        verbose=0
    )

    class_index = np.argmax(
        prediction
    )

    confidence = float(
        np.max(prediction)
    )

    leaf_name = LABEL_ENCODER.inverse_transform(
        [class_index]
    )[0]

    return leaf_name, confidence