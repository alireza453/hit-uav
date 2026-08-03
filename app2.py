import os
import shutil
import tempfile
from pathlib import Path
import streamlit as st
from ultralytics import YOLO

# ==========================================
# Config
# ==========================================

st.set_page_config(
    page_title="YOLO PT Detector",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 YOLO PT Detection")

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Model Selection
# ==========================================
uploaded_model = st.file_uploader(
    "Upload YOLO Model (.pt)",
    type=["pt"]
)

if uploaded_model is None:
    st.info("Please upload a YOLO model.")
    st.stop()

# Save uploaded model temporarily
tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
tmp_model.write(uploaded_model.read())
tmp_model.close()

# Load model
model = YOLO(tmp_model.name)


# ==========================================
# Upload
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"],
)

if uploaded_file is None:
    st.stop()

suffix = Path(uploaded_file.name).suffix.lower()

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tmp.write(uploaded_file.read())
tmp.close()

input_path = tmp.name

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
# Run
# ==========================================

if st.button("🚀 Run Detection", use_container_width=True):

    with st.spinner("Running YOLO..."):

        results = model.predict(
            source=input_path,
            save=True,
            project=OUTPUT_DIR,
            name="predict",
            exist_ok=True,
            verbose=False,
        )

    result_dir = Path(results[0].save_dir)

    output_file = result_dir / Path(input_path).name

    if not output_file.exists():
        files = list(result_dir.iterdir())
        if files:
            output_file = files[0]

    if suffix in [".jpg", ".jpeg", ".png", ".bmp"]:

        output_placeholder.image(
            str(output_file),
            use_container_width=True,
        )

        with open(output_file, "rb") as f:
            st.download_button(
                "Download Result",
                f,
                file_name="result.jpg",
                mime="image/jpeg",
            )

    else:

        output_placeholder.video(str(output_file))

        with open(output_file, "rb") as f:
            st.download_button(
                "Download Result",
                f,
                file_name="result.mp4",
                mime="video/mp4",
            )

    st.success("Detection completed.")