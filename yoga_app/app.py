import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from PoseClassification.pose_embedding import FullBodyPoseEmbedding
from PoseClassification.pose_classifier import PoseClassifier
from PoseClassification.utils import EMADictSmoothing, RepetitionCounter
from PoseClassification.visualize import PoseClassificationVisualizer
import tempfile
import subprocess
import os

class PoseClassifierTransformer(VideoTransformerBase):
    def __init__(self):
        self.fixed_width = 640
        self.fixed_height = 480
        self.pose_tracker = mp_pose.Pose()
        self.pose_embedder = FullBodyPoseEmbedding()
        self.pose_classifier = PoseClassifier(
            pose_samples_folder='yoga_poses',
            pose_embedder=self.pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10
        )
        self.pose_classification_filter = EMADictSmoothing(window_size=10, alpha=0.2)
        self.pose_classification_visualizer = PoseClassificationVisualizer(
            plot_x_max=3000, plot_y_max=10
        )
        self.video_fps = 20
        self.pose_classification_visualizer._fps = self.video_fps
        self.counter = 0

    def transform(self, frame):
        self.counter += 1
        image = frame.to_ndarray(format="bgr24")
        image = cv2.resize(image, (self.fixed_width, self.fixed_height))
        input_frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.counter > 30*5:  # 5 seconds at 30 fps
            result = self.pose_tracker.process(image=input_frame_rgb)
            pose_landmarks_proto = result.pose_landmarks
        else:
            pose_landmarks_proto = None

        # result = self.pose_tracker.process(image=input_frame_rgb)
        # pose_landmarks_proto = result.pose_landmarks

        output_frame = input_frame_rgb.copy()
        pose_classification = None
        pose_classification_filtered = {}

        if pose_landmarks_proto:
            mp_drawing.draw_landmarks(output_frame, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS)
            frame_height, frame_width = output_frame.shape[:2]
            pose_landmarks = np.array([
                [lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                for lmk in pose_landmarks_proto.landmark
            ], dtype=np.float32)

            if pose_landmarks.shape == (33, 3):
                pose_classification = self.pose_classifier(pose_landmarks)
                pose_classification_filtered = self.pose_classification_filter(pose_classification)
                top_class_name = max(pose_classification_filtered, key=pose_classification_filtered.get)
            else:
                top_class_name = None
        else:
            top_class_name = None
            pose_classification_filtered = self.pose_classification_filter(dict())

        # Visualize classification
        output_frame = self.pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=None,
            top_class_name=top_class_name
        )

        output_frame_bgr = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)
        return output_frame_bgr

def process_video(input_video_path, output_video_path='result_raw.mp4'):
    video_cap = cv2.VideoCapture(input_video_path)

    class_name = 'None'
    video_fps = video_cap.get(cv2.CAP_PROP_FPS) or 20
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write to intermediate raw MP4
    out_video = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        video_fps,
        (video_width, video_height)
    )

    # Initialize your pose processing pipeline
    pose_tracker = mp_pose.Pose()
    pose_embedder = FullBodyPoseEmbedding()
    pose_classifier = PoseClassifier(
        pose_samples_folder='yoga_poses',
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10
    )
    pose_classification_filter = EMADictSmoothing(window_size=10, alpha=0.2)
    pose_classification_visualizer = PoseClassificationVisualizer(plot_x_max=3000, plot_y_max=10)
    pose_classification_visualizer._fps = video_fps

    while True:
        success, input_frame = video_cap.read()
        if not success:
            break
        
        input_frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame_rgb)
        pose_landmarks = result.pose_landmarks

        output_frame = input_frame_rgb.copy()
        if pose_landmarks:
            mp_drawing.draw_landmarks(output_frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame_height, frame_width = output_frame.shape[:2]
            landmarks = np.array([
                [lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                for lmk in pose_landmarks.landmark
            ], dtype=np.float32)

            if landmarks.shape == (33, 3):
                pose_classification = pose_classifier(landmarks)
                pose_classification_filtered = pose_classification_filter(pose_classification)
                top_class_name = max(pose_classification_filtered, key=pose_classification_filtered.get)
            else:
                pose_classification = None
                pose_classification_filtered = pose_classification_filter(dict())
                top_class_name = None
        else:
            pose_classification = None
            pose_classification_filtered = pose_classification_filter(dict())
            top_class_name = None

        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=None,
            top_class_name=top_class_name
        )

        output_frame_bgr = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)
        output_frame_bgr = cv2.resize(output_frame_bgr, (video_width, video_height))
        out_video.write(output_frame_bgr)

    video_cap.release()
    out_video.release()
    pose_tracker.close()

    # Re-encode with ffmpeg to ensure browser compatibility
    final_output_path = "result_final.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", output_video_path,
        "-vcodec", "libx264",
        "-movflags", "+faststart",
        "-preset", "ultrafast",
        final_output_path
    ])

    return final_output_path

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Pose Classification")


option = st.radio("Choose input source:", ("Webcam (Live)", "Upload a video"))

col1, col2 = st.columns([5, 3])  # Wide left for video, narrow right for posture examples

with col1:
    if option == "Webcam (Live)":
        st.subheader("Live Pose Classification via Webcam")
        webrtc_streamer(
            key="live-pose",
            video_transformer_factory=PoseClassifierTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True
        )

    elif option == "Upload a video":
        st.subheader("Pose Classification from Uploaded Video")
        video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video-uploader")

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            video_placeholder = st.empty()

            if st.button("Run Pose Classification"):
                with st.spinner("Processing video..."):
                    output_path = process_video(tfile.name)
                    st.success("Processing complete!")
                    video_placeholder.video(output_path)
                    with open(output_path, 'rb') as f:
                        st.download_button("Download Result", f, file_name="classified_result.mp4")

with col2:
    st.markdown("### Possible Postures")

    image_names = [
        "tree.png", "warrior.png", 'downdog.png', 'goddes.png'
    ]

    # Split images into two groups
    row1, row2 = image_names[:2], image_names[2:]

    # Row 1
    cols1 = st.columns(3)
    for col, img_name in zip(cols1, row1):
        full_path = os.path.join("pose_samples_examples", img_name)
        with col:
            if os.path.exists(full_path):
                st.image(full_path, use_container_width=True)
            else:
                st.warning(f"Not found: {img_name}")

    # Row 2
    cols2 = st.columns(3)
    for col, img_name in zip(cols2, row2):
        full_path = os.path.join("pose_samples_examples", img_name)
        with col:
            if os.path.exists(full_path):
                st.image(full_path, use_container_width=True)
            else:
                st.warning(f"Not found: {img_name}")


# if option == "Webcam (Live)":
#     st.subheader("Live Pose Classification via Webcam")
#     webrtc_streamer(
#         key="live-pose",
#         video_transformer_factory=PoseClassifierTransformer,
#         media_stream_constraints={"video": True, "audio": False},
#         async_transform=True
#     )

# elif option == "Upload a video":
#     st.subheader("Pose Classification from Uploaded Video")

#     video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video-uploader")

#     if video_file is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(video_file.read())

#         video_placeholder = st.empty()  # Placeholder to show the video after processing


#         if st.button("Run Pose Classification"):
#             with st.spinner("Processing video..."):
#                 output_path = process_video(tfile.name)
#                 # output_path = process_video(final_output_path)
                
#                 st.success("Processing complete!")
#                 with open('result_final.mp4', 'rb') as video_file:
#                     video_bytes = video_file.read()
#                     st.video(video_bytes)
#                 # video_placeholder.video(output_path)  # Show the result video immediately

#                 with open(output_path, 'rb') as f:
#                     st.download_button("Download Result", f, file_name="classified_result.mp4")
