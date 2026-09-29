import av
import cv2

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

from utils.predictor import predict_leaf


# WEBCAM VIDEO PROCESSOR
class LeafVideoProcessor(
    VideoProcessorBase
):

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        leaf_name, confidence = predict_leaf(
            rgb
        )

        if leaf_name is not None:

            text = (
                f"{leaf_name} "
                f"({confidence * 100:.2f}%)"
            )

            cv2.rectangle(
                img,
                (10, 10),
                (450, 65),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                text,
                (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# START WEBCAM
def start_webcam():

    webrtc_streamer(
        key="leaf-webcam",
        video_processor_factory=LeafVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )