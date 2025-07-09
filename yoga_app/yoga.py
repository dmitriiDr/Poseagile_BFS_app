import argparse
import cv2
import numpy as np
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from PoseClassification.pose_embedding import FullBodyPoseEmbedding
from PoseClassification.pose_classifier import PoseClassifier
from PoseClassification.utils import EMADictSmoothing, RepetitionCounter
from PoseClassification.visualize import PoseClassificationVisualizer

def main(video_path=None, output_path='result.mp4'):
    video_cap = cv2.VideoCapture(video_path if video_path else 0)

    try:
        class_name = 'None'

        video_fps = 20
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

        pose_samples_folder = 'yoga_app/yoga_poses'
        pose_tracker = mp_pose.Pose()
        pose_embedder = FullBodyPoseEmbedding()
        pose_classifier = PoseClassifier(
            pose_samples_folder=pose_samples_folder,
            pose_embedder=pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)
        pose_classification_filter = EMADictSmoothing(window_size=10, alpha=0.2)
        repetition_counter = RepetitionCounter(class_name=class_name, enter_threshold=6, exit_threshold=4)
        pose_classification_visualizer = PoseClassificationVisualizer(plot_x_max=3000, plot_y_max=10)

        frame_idx = 0

        while True:
            success, input_frame = video_cap.read()
            if not success:
                print("Unable to read frame from source.")
                break

            input_frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=input_frame_rgb)
            pose_landmarks = result.pose_landmarks

            output_frame = input_frame_rgb.copy()
            if pose_landmarks:
                mp_drawing.draw_landmarks(output_frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)

                frame_height, frame_width = output_frame.shape[:2]
                pose_landmarks = np.array([
                    [lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                    for lmk in pose_landmarks.landmark
                ], dtype=np.float32)

                if pose_landmarks.shape != (33, 3):
                    print(f"Unexpected landmark shape: {pose_landmarks.shape}")
                    continue

                pose_classification = pose_classifier(pose_landmarks)
                pose_classification_filtered = pose_classification_filter(pose_classification)
                top_class_name = max(pose_classification_filtered, key=pose_classification_filtered.get) if pose_classification_filtered else None
                repetitions_count = repetition_counter(pose_classification_filtered)
            else:
                pose_classification = None
                pose_classification_filtered = pose_classification_filter(dict())
                repetitions_count = repetition_counter.n_repeats
                top_class_name = None

            output_frame = pose_classification_visualizer(
                frame=output_frame,
                pose_classification=pose_classification,
                pose_classification_filtered=pose_classification_filtered,
                repetitions_count=repetitions_count,
                top_class_name=top_class_name)

            pose_classification_visualizer._fps = video_fps

            output_frame_np = np.array(output_frame)
            output_frame_bgr = cv2.cvtColor(output_frame_np, cv2.COLOR_RGB2BGR)
            out_video.write(output_frame_bgr)

            cv2.imshow('Press Q for termination', output_frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    finally:
        video_cap.release()
        out_video.release()
        pose_tracker.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Classification with MediaPipe and Custom Model")
    parser.add_argument('--video', type=str, default=None, help='Path to input video file. If not set, webcam is used.')
    parser.add_argument('--output', type=str, default='result.mp4', help='Path to save the output video.')
    args = parser.parse_args()

    main(video_path=args.video, output_path=args.output)