# import cv2
# import mediapipe as mp #microsoft 
# import pandas as pd

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils

# def extract_keypoints(video_path, output_video_path="static/output.mp4", max_frames=200):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return {"error": "❌ Video file not found or could not be opened."}

#     print("✅ Video opened successfully!")

#     # Get video properties
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Define VideoWriter to save output video
#     fourcc = cv2.VideoWriter_fourcc(*'avc1')  # More compatible codec
#   # Use 'mp4v' for MP4
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     keypoints_data = []
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret or frame_count >= max_frames:
#             break

#         # Convert frame to RGB for MediaPipe
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)

#         if results.pose_landmarks:
#             row = []
#             h, w, _ = frame.shape  

#             # Extract keypoints
#             for landmark in results.pose_landmarks.landmark:
#                 row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
#             keypoints_data.append(row)

#             # Draw keypoints on frame
#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         # Write the processed frame to output video
#         out.write(frame)
#         frame_count += 1

#     cap.release()
#     out.release()  # Save processed video

#     if not keypoints_data:
#         return {"error": "🔴 No keypoints detected in video."}

#     # Create DataFrame with column names
#     columns = [f"K{i}_{c}" for i in range(33) for c in ("x", "y", "z", "visibility")]
#     keypoints_df = pd.DataFrame(keypoints_data, columns=columns)

#     return keypoints_df  # Returns the DataFrame for prediction


import cv2
import mediapipe as mp
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(video_path, output_video_path=None, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "❌ Video file not found or could not be opened."}

    print("✅ Video opened successfully!")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Safe VideoWriter (optional only if output path is given)
    out = None
    if output_video_path:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            print("✅ VideoWriter initialized.")
        except Exception as e:
            print(f"⚠️ Failed to initialize VideoWriter: {e}")
            out = None

    keypoints_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            row = []
            for landmark in results.pose_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            keypoints_data.append(row)

            if out:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if out:
            out.write(frame)

        frame_count += 1

    cap.release()
    if out:
        out.release()

    if not keypoints_data:
        return {"error": "🔴 No keypoints detected in video."}

    columns = [f"K{i}_{c}" for i in range(33) for c in ("x", "y", "z", "visibility")]
    return pd.DataFrame(keypoints_data, columns=columns)
