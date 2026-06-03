# Util functions & Proprocessing

import configs as cfg

# ==============================
# CHECK FOR DEVICE (GPU/CPU)
# ==============================

def check_device():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # Check for NVIDIA GPU, Apple Silicon GPU, or default to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

# ==============================
# PREPROCESSING FUNCTIONS
# ==============================

from pathlib import Path
import shutil

def rename_and_move_imgs(src_dir, dst_dir, fname, prefix="img"):
    """
    Takes a given directory of images and renames them for consistency

    @param src_dir: source directory containing the images
    @param dst_dir: destination directory to save the renamed images
    @param fname: name of the mosquito species
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in img_exts])

    for i, file in enumerate(files):
        new_name = f"{fname}_{i:03d}{file.suffix.lower()}"
        new_path = dst_dir / new_name

        shutil.copy2(file, new_path)  # use shutil.move() if you want to move instead of copy
        print(f"{file.name} -> {new_name}")

if __name__ == "__main__":
    # Rename & Move Images
    fname = "Psorophora_Ciliata"
    src = f'{cfg.DATA_DIR}/{fname}'
    dst = f'{cfg.DATA_DIR}/processed/{fname}'
    rename_and_move_imgs(src, dst, fname)