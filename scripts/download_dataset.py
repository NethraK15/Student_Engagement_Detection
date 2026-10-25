"""
Automated Dataset Downloader for Student Engagement Detection Framework.
Downloads and extracts the preprocessed video frames, OpenFace CSV features, and labels.
"""

import os
import sys
import zipfile

def download_and_extract(file_id=None, download_url=None, destination_dir="dataset"):
    os.makedirs(destination_dir, exist_ok=True)
    archive_path = os.path.join(destination_dir, "dataset_bundle.zip")

    print(">>> Student Engagement Detection - Dataset Fetcher")
    print(f">>> Target extraction folder: {destination_dir}")

    # Instructions if URL is not configured
    if not file_id and not download_url:
        print("\n[!] NOTICE: Cloud dataset URL / Drive ID not yet set in scripts/download_dataset.py.")
        print("Please replace YOUR_DRIVE_FILE_ID or direct URL with your team's dataset link.")
        print("Manual download instructions are available in: data/README.md\n")
        return

    try:
        import gdown
        print(f">>> Downloading dataset archive from Google Drive ID: {file_id} ...")
        gdown.download(id=file_id, output=archive_path, quiet=False)
        
        print(">>> Extracting files ...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)
            
        print(">>> Dataset successfully extracted.")
        if os.path.exists(archive_path):
            os.remove(archive_path)
            print(">>> Temporary archive cleaned up.")

    except ImportError:
        print("[!] Error: 'gdown' package not found. Please install it using:")
        print("    pip install gdown")
    except Exception as e:
        print(f"[!] Download failed: {e}")

if __name__ == "__main__":
    # Team can put their Google Drive File ID here
    DRIVE_FILE_ID = ""  # e.g., '1a2b3c4d5e6f...'
    download_and_extract(file_id=DRIVE_FILE_ID)
