import os
import tempfile
from pathlib import Path
import streamlit as st
from detector import YOLOv8ONNX

# ==========================================
# Streamlit Config
# ==========================================

st.set_page_config(
    page_title="YOLO ONNX Detector",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 YOLO ONNX Detection")

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Class Names
# ==========================================

CLASS_NAMES =  ["Person", "Car", "Bicycle", "OtherVehicle", "DontCare"]

# ==========================================
# Load Models
# ==========================================

model_files = sorted(Path(MODEL_DIR).glob("*.onnx"))

if len(model_files) == 0:
    st.error("No ONNX model found in models folder.")
    st.stop()

selected_model = st.selectbox(
    "Select ONNX Model",
    [m.name for m in model_files]
)


@st.cache_resource
def load_model(model_path):

    return YOLOv8ONNX(
        model_path="models/best.onnx",
        class_names=CLASS_NAMES,
    )


detector = load_model(
    os.path.join(MODEL_DIR, selected_model)
)

# ==========================================
# Upload
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "mp4",
        "avi",
        "mov"
    ]
)

if uploaded_file is None:
    st.stop()

suffix = Path(uploaded_file.name).suffix.lower()

temp_file = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix
)

temp_file.write(uploaded_file.read())
temp_file.close()

input_path = temp_file.name

# ==========================================
# Preview
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Input")

    if suffix in [".jpg", ".jpeg", ".png", ".bmp"]:

        st.image(input_path, use_container_width=True)

    else:

        st.video(input_path)

with col2:

    st.subheader("Output")

    output_placeholder = st.empty()

# ==========================================
# Run Button
# ==========================================

if st.button("🚀 Run Detection", use_container_width=True):

    output_path = os.path.join(
        OUTPUT_DIR,
        f"result{suffix}"
    )

    progress = st.progress(0)

    status = st.empty()

    # ======================================
    # IMAGE
    # ======================================

    if suffix in [".jpg", ".jpeg", ".png", ".bmp"]:

        status.text("Running inference...")

        detector.detect_image(
            input_path,
            output_path
        )

        progress.progress(100)

        output_placeholder.image(
            output_path,
            use_container_width=True
        )

        st.success("Finished")

        with open(output_path, "rb") as f:

            st.download_button(
                "Download Result",
                f,
                file_name="result.jpg",
                mime="image/jpeg"
            )

    # ======================================
    # VIDEO
    # ======================================

    else:

        def update(current, total):

            progress.progress(current / total)

            status.text(
                f"Processing frame {current}/{total}"
            )

        detector.detect_video(
            input_path,
            output_path,
            progress_callback=update
        )

        output_placeholder.video(output_path)

        st.success("Finished")

        with open(output_path, "rb") as f:

            st.download_button(
                "Download Result",
                f,
                file_name="result.mp4",
                mime="video/mp4"
            )

    progress.empty()

    status.empty()