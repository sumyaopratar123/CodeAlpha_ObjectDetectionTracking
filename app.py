import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 17px;
    margin-bottom: 30px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background: #1f2937;
    margin-bottom: 15px;
}

.stButton button {
    width: 100%;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">🎯 AI Object Detection & Tracking</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">YOLO-powered real-time object detection with tracking IDs</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.40,
    step=0.05
)

tracker = st.sidebar.selectbox(
    "Tracking Algorithm",
    ["ByteTrack", "BoT-SORT"]
)

source_type = st.sidebar.radio(
    "Input Source",
    ["📹 Upload Video", "📷 Webcam"]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Technology Stack**

    • Python  
    • OpenCV  
    • YOLO  
    • ByteTrack / BoT-SORT  
    • Streamlit
    """
)

# --------------------------------------------------
# UPLOAD VIDEO
# --------------------------------------------------

if source_type == "📹 Upload Video":

    st.subheader("📹 Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:

        # Save uploaded video temporarily
        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(uploaded_file.read())
        temp_input.close()

        st.success("Video uploaded successfully!")

        if st.button("🚀 Start Detection & Tracking"):

            cap = cv2.VideoCapture(temp_input.name)

            if not cap.isOpened():
                st.error("Could not open video.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_path = output_file.name
            output_file.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            frame_placeholder = st.empty()

            progress = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = 0

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # ------------------------------------------
                # OBJECT DETECTION + TRACKING
                # ------------------------------------------

                results = model.track(
                    frame,
                    persist=True,
                    conf=confidence,
                    tracker="bytetrack.yaml"
                    if tracker == "ByteTrack"
                    else "botsort.yaml",
                    verbose=False
                )

                annotated_frame = results[0].plot()

                # ------------------------------------------
                # TRACKING INFORMATION
                # ------------------------------------------

                boxes = results[0].boxes

                object_count = 0

                if boxes is not None:

                    object_count = len(boxes)

                    if boxes.id is not None:

                        track_ids = boxes.id.int().cpu().tolist()

                        for track_id in track_ids:

                            cv2.putText(
                                annotated_frame,
                                f"ID: {track_id}",
                                (20, 40 + (track_id % 10) * 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )

                # ------------------------------------------
                # DISPLAY COUNT
                # ------------------------------------------

                cv2.putText(
                    annotated_frame,
                    f"Objects: {object_count}",
                    (20, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                # Write output
                writer.write(annotated_frame)

                # Streamlit uses RGB
                rgb_frame = cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_BGR2RGB
                )

                frame_placeholder.image(
                    rgb_frame,
                    channels="RGB",
                    use_container_width=True
                )

                if total_frames > 0:
                    progress.progress(
                        min(frame_count / total_frames, 1.0)
                    )

            cap.release()
            writer.release()

            progress.progress(1.0)

            st.success("🎉 Detection and tracking completed!")

            # ------------------------------------------
            # DOWNLOAD RESULT
            # ------------------------------------------

            with open(output_path, "rb") as file:

                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=file,
                    file_name="object_detection_tracking.mp4",
                    mime="video/mp4"
                )

            # Cleanup
            try:
                os.remove(temp_input.name)
            except:
                pass


# --------------------------------------------------
# WEBCAM MODE
# --------------------------------------------------

else:

    st.subheader("📷 Webcam Detection")

    st.warning(
        "For local testing, webcam access works through OpenCV. "
        "Press 'Start Webcam' and allow camera access if Windows asks."
    )

    start_webcam = st.button("▶️ Start Webcam")

    stop_webcam = st.button("⏹️ Stop Webcam")

    if start_webcam:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            st.error(
                "Could not access webcam. "
                "Check camera permissions or whether another app is using the camera."
            )

            st.stop()

        frame_placeholder = st.empty()

        stop_placeholder = st.empty()

        while True:

            ret, frame = cap.read()

            if not ret:
                st.error("Failed to read webcam frame.")
                break

            # Detection + Tracking
            results = model.track(
                frame,
                persist=True,
                conf=confidence,
                tracker="bytetrack.yaml"
                if tracker == "ByteTrack"
                else "botsort.yaml",
                verbose=False
            )

            annotated_frame = results[0].plot()

            boxes = results[0].boxes

            object_count = 0

            if boxes is not None:

                object_count = len(boxes)

                if boxes.id is not None:

                    track_ids = boxes.id.int().cpu().tolist()

                    for track_id in track_ids:

                        cv2.putText(
                            annotated_frame,
                            f"ID: {track_id}",
                            (
                                20,
                                40 + (track_id % 10) * 30
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

            cv2.putText(
                annotated_frame,
                f"Objects: {object_count}",
                (20, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            rgb_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                rgb_frame,
                channels="RGB",
                use_container_width=True
            )

            # Stop condition
            if stop_webcam:
                break

        cap.release()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "AI Object Detection & Tracking | YOLO + OpenCV + Streamlit"
)