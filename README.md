# UAV Thermal Object Detection with YOLO

![UAV Thermal Detection Demo](https://github.com/alireza453/hit-uav/blob/main/outputs/60m-40_3.gif)

A deep learning-based object detection system for detecting and localizing objects from **UAV thermal imagery** using YOLO models.

The project supports inference on thermal images and videos captured by drones, enabling detection of targets such as:

- Person
- Car
- Bicycle
- OtherVehicle
- DontCare

The system provides a simple Streamlit interface for uploading YOLO models, images, and videos, then visualizing detection results with bounding boxes.

---

## Features

✅ YOLO `.pt` model support  
✅ Thermal image object detection  
✅ Thermal video processing  
✅ Image and video upload  
✅ Automatic bounding box visualization  
✅ Confidence score display  
✅ Streamlit web interface  
✅ GPU acceleration support (CUDA)

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-name/UAV-Thermal-Detection.git

cd HIT-UAV
```

## 2. Create Virtual env and activate it

```bash
python -m venv <NAME>
```

## 3. Run

```bash
streamlit run app.py
```

- app.py uses onnx model
- app2.py uses pt model
