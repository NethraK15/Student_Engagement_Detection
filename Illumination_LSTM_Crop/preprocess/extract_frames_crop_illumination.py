import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Add parent directory to sys.path to import config_exp6
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_exp6

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.3gp', '.flv', '.webm')

# Try importing MediaPipe for modern neural face detection
try:
    import mediapipe as mp
    MP_FACE_DETECTION = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.3
    )
except Exception:
    MP_FACE_DETECTION = None

# Initialize OpenCV built-in Haar Cascade Face Detector as secondary backup
try:
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    FACE_CASCADE = None

def detect_and_crop_face(frame_bgr, last_box=None, margin_ratio=0.25):
    """
    Detects the primary student face in the video frame, expands the bounding box by margin_ratio,
    and crops out background objects.
    Pipeline: MediaPipe Neural Detector -> OpenCV Haar Cascade -> Last Valid Face Box -> Upper-Head Crop.
    Returns: (crop_img, bbox_tuple, is_real_detection_flag)
    """
    img_h, img_w = frame_bgr.shape[:2]

    # --- 1. Primary Detector: MediaPipe (Modern Neural Face Detector) ---
    if MP_FACE_DETECTION is not None:
        try:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = MP_FACE_DETECTION.process(rgb_frame)
            if results.detections:
                best_detection = max(results.detections, key=lambda d: d.score[0])
                bboxC = best_detection.location_data.relative_bounding_box
                
                x = int(max(0, bboxC.xmin * img_w))
                y = int(max(0, bboxC.ymin * img_h))
                w = int(bboxC.width * img_w)
                h = int(bboxC.height * img_h)

                if w > 10 and h > 10:
                    margin_w = int(w * margin_ratio)
                    margin_h = int(h * margin_ratio)

                    x1 = max(0, x - margin_w)
                    y1 = max(0, y - margin_h)
                    x2 = min(img_w, x + w + margin_w)
                    y2 = min(img_h, y + h + margin_h)

                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        return crop, (x1, y1, x2, y2), True
        except Exception:
            pass

    # --- 2. Secondary Detector: OpenCV Haar Cascade ---
    if FACE_CASCADE is not None and not FACE_CASCADE.empty():
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            margin_w = int(w * margin_ratio)
            margin_h = int(h * margin_ratio)

            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(img_w, x + w + margin_w)
            y2 = min(img_h, y + h + margin_h)

            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                return crop, (x1, y1, x2, y2), True

    # --- 3. Fallback A: Use last known REAL face bounding box ---
    if last_box is not None:
        x1, y1, x2, y2 = last_box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            return crop, last_box, False

    # --- 4. Fallback B: Upper-Head Focus Crop ---
    # Focus on top 60% of frame where head & face are located (avoids chest/torso)
    box_size = int(min(img_w, img_h * 0.8))
    start_x = max(0, (img_w - box_size) // 2)
    start_y = 0  # Start from top of frame (y=0) to guarantee capturing head/eyes
    x1, y1, x2, y2 = start_x, start_y, start_x + box_size, start_y + box_size
    
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2), False

def apply_clahe(img_rgb):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the L-channel of the LAB color space.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)

def apply_sharpen(img_rgb):
    """
    Applies 3x3 Laplacian edge sharpening kernel to enhance facial contour boundaries.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(img_rgb.astype(np.float32), -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def apply_illumination_enhancement(img_rgb):
    """
    Combined illumination normalization pipeline: CLAHE + Laplacian Sharpening.
    """
    return apply_sharpen(apply_clahe(img_rgb))

def find_video_files(dataset_dir, classes):
    """
    Scans the dataset directory and lists all video files.
    Matches folder names to classes robustly using normalization:
    - contains 'disengaged' -> Disengaged (label 2)
    - contains 'partially', 'partial', or 'enaged' -> Partially Engaged (label 1)
    - contains 'engaged' (but not dis/part) -> Engaged (label 0)
    """
    video_list = []
    if not os.path.exists(dataset_dir):
        print(f"[Error] Dataset directory not found: {dataset_dir}")
        return video_list

    dir_contents = os.listdir(dataset_dir)
    matched_folders = {0: [], 1: [], 2: []}
    
    for item in dir_contents:
        item_path = os.path.join(dataset_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        norm_name = item.lower().replace(" ", "").replace("_", "").replace("-", "")
        
        if "disengaged" in norm_name:
            matched_folders[2].append(item)
        elif "partially" in norm_name or "partial" in norm_name or "enaged" in norm_name:
            matched_folders[1].append(item)
        elif "engaged" in norm_name:
            matched_folders[0].append(item)
            
    for label_idx in range(3):
        folders = matched_folders[label_idx]
        cls = classes[label_idx]
        if not folders:
            print(f"[Warning] No directory matched for class '{cls}' in {dataset_dir}")
            continue
            
        for match_folder in folders:
            print(f"[Info] Mapping folder '{match_folder}' to class '{cls}' (label {label_idx})")
            class_dir = os.path.join(dataset_dir, match_folder)
            for root, _, files in os.walk(class_dir):
                for file in sorted(files):
                    if file.lower().endswith(VIDEO_EXTENSIONS):
                        video_list.append({
                            "video_path": os.path.join(root, file),
                            "class_name": cls,
                            "label": label_idx,
                            "filename": file,
                            "base_name": os.path.splitext(file)[0]
                        })
    return video_list

def extract_and_illuminate_frames(video_info, output_dir, target_size=config_exp6.IMAGE_SIZE):
    """
    Detects & crops student face, resizes to target_size, applies CLAHE + Sharpening,
    and saves as a .npy file (with local buffer to bypass GDrive FUSE write latency).
    """
    video_path = video_info["video_path"]
    class_name = video_info["class_name"]
    base_name = video_info["base_name"]
    
    save_filename = f"{class_name}__{base_name}.npy"
    save_path = os.path.join(output_dir, save_filename)
    
    # Resume check
    if os.path.exists(save_path):
        return True, "Already processed (skipped)"
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video file"
        
    frames = []
    last_box = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Fast detect and crop student face (removes background) with MediaPipe / Haar Cascade
        face_crop_bgr, box, is_real = detect_and_crop_face(frame, last_box=last_box)
        if is_real:
            last_box = box
        
        # 2. Resize face crop to 224x224
        resized_crop = cv2.resize(face_crop_bgr, target_size)
        
        # 3. Convert BGR to RGB
        rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
        
        # 4. Apply Illumination Pipeline (CLAHE + Sharpening) on the face crop
        enhanced_face = apply_illumination_enhancement(rgb_crop)
        frames.append(enhanced_face)
    
    cap.release()
    
    if len(frames) == 0:
        return False, "0 frames extracted"
        
    frames_np = np.array(frames, dtype=np.uint8) # Shape: [N, H, W, 3]

    # Fast I/O buffering: if target path is in Google Drive, save locally first then move
    import shutil
    if save_path.startswith("/content/drive/"):
        temp_dir = "/content/temp_exp6_frames"
        os.makedirs(temp_dir, exist_ok=True)
        local_path = os.path.join(temp_dir, save_filename)
        np.save(local_path, frames_np)
        shutil.move(local_path, save_path)
    else:
        np.save(save_path, frames_np)

    return True, f"Success ({len(frames)} illuminated face crops)"

def main():
    print("==================================================")
    print("  EXPERIMENT 6: FACE CROPPING & ILLUMINATION      ")
    print("     (CLAHE + SHARPENING) TO GOOGLE DRIVE         ")
    print("==================================================")
    print(f"Dataset root : {config_exp6.DATASET_DIR}")
    print(f"Classes      : {config_exp6.CLASSES}")
    print(f"Output dir   : {config_exp6.PREPROCESSED_FRAMES_DIR}\n")

    os.makedirs(config_exp6.PREPROCESSED_FRAMES_DIR, exist_ok=True)

    videos = find_video_files(config_exp6.DATASET_DIR, config_exp6.CLASSES)
    if not videos:
        print("[Error] No videos found in the dataset directory. Check paths in config_exp6.py.")
        sys.exit(1)
        
    print(f"Found {len(videos)} videos to process.")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for video in tqdm(videos, desc="Cropping faces & illuminating"):
        success, message = extract_and_illuminate_frames(video, config_exp6.PREPROCESSED_FRAMES_DIR)
        if success:
            if "skipped" in message:
                skip_count += 1
            else:
                success_count += 1
        else:
            print(f"\n[Warning] Failed to process '{video['filename']}': {message}")
            error_count += 1
            
    print("\n" + "=" * 50)
    print("  FACE CROPPING & ILLUMINATION COMPLETE")
    print("=" * 50)
    print(f"Newly processed: {success_count}")
    print(f"Resumed/Skipped: {skip_count}")
    print(f"Failed         : {error_count}")
    print(f"Total files in output: {len(os.listdir(config_exp6.PREPROCESSED_FRAMES_DIR))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
