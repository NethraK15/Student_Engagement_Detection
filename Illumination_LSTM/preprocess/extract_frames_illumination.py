import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Add parent directory to sys.path to import config_exp4
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_exp4

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.3gp', '.flv', '.webm')

# Initialize OpenCV built-in Haar Cascade Face Detector
try:
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    FACE_CASCADE = None

def detect_and_crop_face(frame_bgr, margin_ratio=0.2):
    """
    Detects the primary face in the video frame, expands the bounding box by margin_ratio
    to preserve facial features, and crops out background objects.
    Falls back to a center-square crop if no face is detected in the frame.
    """
    if FACE_CASCADE is not None and not FACE_CASCADE.empty():
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        if len(faces) > 0:
            # Select the largest detected face box by area
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Add margin around face box
            margin_w = int(w * margin_ratio)
            margin_h = int(h * margin_ratio)
            
            img_h, img_w, _ = frame_bgr.shape
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(img_w, x + w + margin_w)
            y2 = min(img_h, y + h + margin_h)
            
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                return crop

    # Fallback if no face detected in frame: center-square crop
    h, w, _ = frame_bgr.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return frame_bgr[start_y:start_y+min_dim, start_x:start_x+min_dim]

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

def extract_and_illuminate_frames(video_info, output_dir, target_size=config_exp4.IMAGE_SIZE):
    """
    Detects & crops student face, resizes to target_size, applies CLAHE + Sharpening,
    and saves as a .npy file directly to Google Drive.
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Detect and crop student face (removes background)
        face_crop_bgr = detect_and_crop_face(frame)
        
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
    np.save(save_path, frames_np)
    return True, f"Success ({len(frames)} illuminated face crops)"

def main():
    print("==================================================")
    print("  EXPERIMENT 4: FACE CROPPING & ILLUMINATION      ")
    print("     (CLAHE + SHARPENING) TO GOOGLE DRIVE         ")
    print("==================================================")
    print(f"Dataset root : {config_exp4.DATASET_DIR}")
    print(f"Classes      : {config_exp4.CLASSES}")
    print(f"Output dir   : {config_exp4.PREPROCESSED_FRAMES_DIR}\n")

    os.makedirs(config_exp4.PREPROCESSED_FRAMES_DIR, exist_ok=True)

    videos = find_video_files(config_exp4.DATASET_DIR, config_exp4.CLASSES)
    if not videos:
        print("[Error] No videos found in the dataset directory. Check paths in config_exp4.py.")
        sys.exit(1)
        
    print(f"Found {len(videos)} videos to process.")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for video in tqdm(videos, desc="Cropping faces & illuminating"):
        success, message = extract_and_illuminate_frames(video, config_exp4.PREPROCESSED_FRAMES_DIR)
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
    print(f"Total files in output: {len(os.listdir(config_exp4.PREPROCESSED_FRAMES_DIR))}")
    print("=" * 50)

if __name__ == "__main__":
    main()
