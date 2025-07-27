import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import emoji

# Set page configuration
st.set_page_config(
    page_title="SHELF-EYE " + emoji.emojize(":eyes:"),
    page_icon="C:\\Users\\bilal\\OneDrive\\Desktop\\FYP\\SHELFEYE.ico",
)

# Load the YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Sidebar settings
st.sidebar.title("⚙️ Model Settings")
model_path = st.sidebar.text_input("📁 Enter YOLO model path:", "C:\\Personal Documents\\FAST\\8th Semester\\FYP\\results\\detect\\train\\weights\\best.pt")
conf_threshold = st.sidebar.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.25)

# Main Interface
st.title("SHELF-EYE " + emoji.emojize(":eyes:"))
st.markdown("### 🔍 Powered by BIMAN Solutions")
st.write("Upload an image or video and run detection to identify empty shelf spaces!")

# File uploader
uploaded_file = st.file_uploader("📷 Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

        # Run detection on image
        if st.button("🚀 Run Detection on Image"):
            model = load_model(model_path)
            results = model.predict(source=image, conf=conf_threshold, save=False, imgsz=416)

            # Display results
            st.write("### 📌 Detected Objects:")
            detections = results[0].boxes.data.cpu().numpy()
            for box in detections:
                x1, y1, x2, y2, conf, class_id = box
                class_name = model.names[int(class_id)]
                st.write(f"✔️ Class: {class_name}, Confidence: {conf:.2f}")

            # Show annotated image
            annotated_image = results[0].plot()
            st.image(annotated_image, caption="📊 Detection Results", use_container_width=True)

    elif "video" in file_type:
        # Handle video processing
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        if st.button("🎬 Run Detection on Video"):
            model = load_model(model_path)

            # Open video file
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Output video writer
            output_path = "output_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run detection
                results = model.predict(source=rgb_frame, conf=conf_threshold, save=False, imgsz=416)

                # Annotate frame
                annotated_frame = results[0].plot()

                # Write frame to output video
                out.write(annotated_frame)

                # Display frame in Streamlit
                stframe.image(annotated_frame, channels="RGB", use_container_width=True)

            cap.release()
            out.release()

            st.success("✅ Video processing complete! Download the output video below.")
            with open(output_path, "rb") as video_file:
                st.download_button(label="⬇️ Download Processed Video", data=video_file, file_name="output_video.mp4", mime="video/mp4")

st.write("🔔 **Note:** Real-time webcam detection will be available in future updates.")