"""
Dataset Downloader for VisionGuard AI.

Downloads target MVTec AD categories (metal_nut, screw, capsule)
directly from the verified Hugging Face mirror into data/raw/.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

TARGET_CATEGORIES = ["metal_nut", "screw", "capsule"]
REPO_ID = "foersben/mvtec-ad"


def download_target_categories(target_dir: str = "data/raw") -> Path:
    """Download the 3 target categories using snapshot_download with pattern filtering."""
    dest_path = Path(target_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print(" VisionGuard AI - Dataset Downloader")
    print(f" Source Repository: {REPO_ID}")
    print(f" Target Categories: {', '.join(TARGET_CATEGORIES)}")
    print(f" Destination:       {dest_path}")
    print("==================================================\n")

    patterns = [f"{cat}/*" for cat in TARGET_CATEGORIES]
    print(f"Starting download for patterns: {patterns}...")

    downloaded_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(dest_path),
        local_dir_use_symlinks=False,
        max_workers=8
    )

    print(f"\nDownload complete! Files saved to: {downloaded_dir}")
    return Path(downloaded_dir)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    download_target_categories(out_dir)

