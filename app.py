import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = load_model("models/digit_recognition_model.h5")

st.title("✍️ Handwritten Digit Recognition")
st.markdown(
    "Draw a digit (0-9) in the canvas below and the CNN model will predict it."
)

        
# ==========================================
# DRAW DIGIT SECTION
# ==========================================

st.markdown("---")
st.header("✏️ Draw a Digit")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    update_streamlit=True,
    key="canvas",
)

if canvas_result.image_data is not None:

    import cv2

    img = canvas_result.image_data[:, :, 0]

    if np.sum(img) > 0:

        img = img.astype("uint8")

        _, thresh = cv2.threshold(
            img,
            50,
            255,
            cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:

            largest_contour = max(
                contours,
                key=cv2.contourArea
            )

            x, y, w, h = cv2.boundingRect(
                largest_contour
            )

            digit = thresh[y:y+h, x:x+w]

            digit = cv2.copyMakeBorder(
                digit,
                20,
                20,
                20,
                20,
                cv2.BORDER_CONSTANT,
                value=0
            )

            digit = cv2.resize(
                digit,
                (28, 28)
            )

            st.image(
                digit,
                caption="Processed Drawing",
                width=150
            )

            digit = digit.astype(
                "float32"
            ) / 255.0

            digit = digit.reshape(
                1,
                28,
                28,
                1
            )

            prediction = model.predict(
                digit,
                verbose=0
            )

            pred = np.argmax(
                prediction
            )

            confidence = (
                np.max(prediction) * 100
            )

            st.success(
                f"Predicted Digit: {pred}"
            )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )    