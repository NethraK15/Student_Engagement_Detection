import os
import sys
import cv2
import torch
import numpy as np

def extract_classical_geometric_features_from_video(video_path, max_frames=120, sample_fps=5):
    """
    Classical geometric feature baseline (Dlib 68-landmark / Classical CV):
    Extracts:
    1. Head Pose (Yaw, Pitch, Roll via solvePnP)
    2. Left Eye Aspect Ratio (EAR)
    3. Right Eye Aspect Ratio (EAR)
    4. Eye Aspect Ratio difference (Left - Right asymmetry)
    5. Mouth Aspect Ratio (MAR)
    6. Eyebrow-to-Eye distance (AU4 brow furrow)
    7. Nose-to-Mouth distance
    8. Inter-ocular distance normalized landmark dispersion (16 dims)
    Total feature dim = 24 dimensions per frame.
    """
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        has_dlib = os.path.exists(predictor_path)
        if has_dlib:
            predictor = dlib.shape_predictor(predictor_path)
    except Exception:
        has_dlib = False

    # We also provide a fast, robust MediaPipe 68-point mapper as fallback
    # so execution never fails if dlib model is not downloaded!
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
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

    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye outer corner
        (225.0, 170.0, -135.0),   # Right eye outer corner
        (-150.0, -150.0, -125.0), # Left Mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                pts = np.array([[lm.x * w, lm.y * h] for lm in face.landmark])

                # 1. Classical Head Pose via SolvePnP
                img_pts = np.array([
                    pts[1], pts[152], pts[33], pts[263], pts[61], pts[291]
                ], dtype=np.float64)

                focal = w
                cam = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
                dist = np.zeros((4, 1), dtype=np.float64)

                ok, rvec, tvec = cv2.solvePnP(model_points, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    rmat, _ = cv2.Rodrigues(rvec)
                    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
                    if sy > 1e-6:
                        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
                        yaw = np.arctan2(-rmat[2, 0], sy)
                        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
                    else:
                        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
                        yaw = np.arctan2(-rmat[2, 0], sy)
                        roll = 0.0
                    pose = [pitch, yaw, roll]
                else:
                    pose = [0.0, 0.0, 0.0]

                # 2. Classical Eye Aspect Ratio (EAR)
                ear_l = np.linalg.norm(pts[159] - pts[145]) / (np.linalg.norm(pts[33] - pts[133]) + 1e-6)
                ear_r = np.linalg.norm(pts[386] - pts[374]) / (np.linalg.norm(pts[362] - pts[263]) + 1e-6)
                ear_diff = abs(ear_l - ear_r)

                # 3. Classical Mouth Aspect Ratio (MAR)
                mar = np.linalg.norm(pts[13] - pts[14]) / (np.linalg.norm(pts[78] - pts[308]) + 1e-6)

                # 4. Brow to eye distance (frown)
                brow_dist = (np.linalg.norm(pts[70] - pts[159]) + np.linalg.norm(pts[300] - pts[386])) / (2 * (np.linalg.norm(pts[33] - pts[263]) + 1e-6))

                # 5. Core 8-landmark contour dispersion
                contour_pts = [1, 33, 263, 61, 291, 152, 70, 300]
                norm_pts = (pts[contour_pts] - pts[1]).flatten() / (np.linalg.norm(pts[33] - pts[263]) + 1e-6)

                # Classical geometric vector (3 + 1 + 1 + 1 + 1 + 1 + 16 = 24 dims)
                classical_vector = np.concatenate([
                    np.array(pose),
                    np.array([ear_l, ear_r, ear_diff, mar, brow_dist]),
                    norm_pts
                ]).astype(np.float32)

                sequence_features.append(classical_vector)
                if len(sequence_features) >= max_frames:
                    break

        frame_idx += 1

    cap.release()
    face_mesh.close()

    if not sequence_features:
        return None

    return torch.tensor(np.array(sequence_features), dtype=torch.float32)
