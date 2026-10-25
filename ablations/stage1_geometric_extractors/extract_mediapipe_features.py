import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm

def get_mediapipe_features_from_video(video_path, max_frames=120, sample_fps=5):
    """
    Extracts geometric features using MediaPipe Face Mesh:
    - 3D Head Pose (Yaw, Pitch, Roll via solvePnP)
    - Left & Right Eye Aspect Ratio (EAR) for blink & gaze fixation
    - Mouth Aspect Ratio (MAR) for speech/yawn
    - Eyebrow-to-Eye distances (AUs: brow raise/furrow)
    - Normalized key 3D landmark displacements (48 coordinates)
    Total feature dim = 3 + 2 + 1 + 2 + 48 = 56 dimensions per frame.
    """
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 30.0

    step = max(1, int(round(orig_fps / sample_fps)))

    frame_idx = 0
    sequence_features = []

    # 3D model points for Head Pose SolvePnP
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
        (0.0, -330.0, -65.0),     # Chin (landmark 152)
        (-225.0, 170.0, -135.0),  # Left eye outer corner (landmark 33)
        (225.0, 170.0, -135.0),   # Right eye outer corner (landmark 263)
        (-150.0, -150.0, -125.0), # Left Mouth corner (landmark 61)
        (150.0, -150.0, -125.0)   # Right mouth corner (landmark 291)
    ], dtype=np.float64)

    # Key landmark indices for geometry
    KEY_LANDMARKS = [
        1, 33, 61, 152, 199, 263, 291, 70, 63, 105, 66, 107, 336, 296, 334, 293
    ] # 16 landmarks * 3 (x, y, z) = 48 coordinates

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face.landmark])

                # 1. Head Pose (SolvePnP)
                image_points = np.array([
                    pts[1][:2], pts[152][:2], pts[33][:2],
                    pts[263][:2], pts[61][:2], pts[291][:2]
                ], dtype=np.float64)

                focal_length = w
                cam_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ], dtype=np.float64)
                dist_coeffs = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(
                    model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    rot_mat, _ = cv2.Rodrigues(rot_vec)
                    # Euler angles
                    sy = np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
                    singular = sy < 1e-6
                    if not singular:
                        pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
                        yaw = np.arctan2(-rot_mat[2, 0], sy)
                        roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
                    else:
                        pitch = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
                        yaw = np.arctan2(-rot_mat[2, 0], sy)
                        roll = 0
                    pose_feats = [pitch, yaw, roll]
                else:
                    pose_feats = [0.0, 0.0, 0.0]

                # 2. Left and Right Eye Aspect Ratio (EAR)
                def get_ear(p_top, p_bot, p_left, p_right):
                    v = np.linalg.norm(pts[p_top][:2] - pts[p_bot][:2])
                    h = np.linalg.norm(pts[p_left][:2] - pts[p_right][:2]) + 1e-6
                    return v / h

                ear_left = get_ear(159, 145, 33, 133)
                ear_right = get_ear(386, 374, 362, 263)

                # 3. Mouth Aspect Ratio (MAR)
                mar = np.linalg.norm(pts[13][:2] - pts[14][:2]) / (np.linalg.norm(pts[78][:2] - pts[308][:2]) + 1e-6)

                # 4. Eyebrow distances (AU1 & AU4 brow furrow/raise)
                brow_left = np.linalg.norm(pts[70][:2] - pts[159][:2]) / (w + 1e-6)
                brow_right = np.linalg.norm(pts[300][:2] - pts[386][:2]) / (w + 1e-6)

                # 5. Key 3D relative coordinates (normalized by bounding box scale)
                key_pts = (pts[KEY_LANDMARKS] - pts[1]).flatten()
                norm_scale = np.linalg.norm(pts[33][:2] - pts[263][:2]) + 1e-6
                norm_key_pts = key_pts / norm_scale

                # Combined vector: 3 + 2 + 1 + 2 + 48 = 56 dims
                frame_vector = np.concatenate([
                    np.array(pose_feats),
                    np.array([ear_left, ear_right, mar, brow_left, brow_right]),
                    norm_key_pts
                ]).astype(np.float32)

                sequence_features.append(frame_vector)

                if len(sequence_features) >= max_frames:
                    break

        frame_idx += 1

    cap.release()
    face_mesh.close()

    if not sequence_features:
        return None

    return torch.tensor(np.array(sequence_features), dtype=torch.float32)
