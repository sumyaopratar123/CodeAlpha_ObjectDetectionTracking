# Task 4 — Object Detection and Tracking

A Streamlit application that accepts a video, runs YOLO object detection with persistent tracking IDs, draws annotated bounding boxes, writes a processed MP4, previews it, and provides a download button.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

On first use, Ultralytics downloads the small `yolo11n.pt` model automatically. An internet connection is required for that first model download unless the model is already cached.
