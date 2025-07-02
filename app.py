import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model
model = load_model("facetracker.h5")

# Streamlit UI
st.title("📸 Face Detection with Webcam")
option = st.radio("Choose input type:", ["Upload Image", "Use Webcam"])

# Helper function to draw bounding box
def draw_bbox(image, coords, confidence):
    h, w = image.shape[:2]
    coords = np.multiply(coords, [w, h, w, h]).astype(int)
    cv2.rectangle(image,
                  tuple(coords[:2]),
                  tuple(coords[2:]),
                  (0, 255, 0), 2)
    cv2.putText(image, f"{confidence:.2f}", tuple(coords[:2]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        original = image.copy()
        image_resized = cv2.resize(image, (120, 120))
        input_image = np.expand_dims(image_resized, axis=0)

        class_pred, box_pred = model.predict(input_image)

        if class_pred[0][0] > 0.9:
            draw_bbox(original, box_pred[0], class_pred[0][0])
            st.success(f"✅ Face detected with confidence: {class_pred[0][0]:.2f}")
        else:
            st.warning("❌ No confident face detected.")

        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Detected Output", use_column_width=True)

elif option == "Use Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    run = st.checkbox("Start Webcam")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        image_resized = cv2.resize(frame, (120, 120))
        input_image = np.expand_dims(image_resized, axis=0)
        class_pred, box_pred = model.predict(input_image)

        if class_pred[0][0] > 0.9:
            draw_bbox(frame, box_pred[0], class_pred[0][0])

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    st.info("Webcam stopped.")
