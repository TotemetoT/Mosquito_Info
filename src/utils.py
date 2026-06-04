# Util functions & Proprocessing

import configs as cfg

# ==============================
# CHECK FOR DEVICE (GPU/CPU)
# ==============================

def check_device():
    import torch

    # Check for NVIDIA GPU, Apple Silicon, or default to CPU
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

def rename_and_move_imgs(src_dir, dst_dir, fname, prefix="img"):
    """
    Takes a given directory of images and renames them for consistency

    @param src_dir: source directory
    @param dst_dir: destination directory
    @param fname: mosquito species
    """

    from pathlib import Path
    import shutil

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in img_exts])

    for i, file in enumerate(files):
        new_name = f"{fname:02d}{i:03d}{file.suffix.lower()}" # 3 digits, can change later
        new_path = dst_dir / new_name

        shutil.copy2(file, new_path)  # use shutil.move() if you want to move instead of copy
        print(f"{file.name} -> {new_name}")

# ======================================
# IMAGE IDENTIFICATION (FILENAME-BASED)
# ======================================

def identify_img(img):
    """
    Identifies mosquito species using file name

    @param img: image file name (string)
    @return tuple: (bool, species name or error message)
    """
    img = str(img)
    if img[0] == "0":
        mosq = int(img[1])
    else:
        mosq = int(img[:2])
    if mosq in cfg.MOSQ_MAP: return (True, cfg.MOSQ_MAP[mosq])
    return (False, "Not in List")

# ==============================
# TEST DATASET.PY 
# ==============================

def test_dataset(split):
    """
    Tests the MosquitoDataset class by Loading a sample split
    Will display a random image from the dataset

    @param split: dataset split to test (train/val/test)
    """
    from dataset import MosquitoDataset

    from torchvision import transforms
    import matplotlib.pyplot as plt
    import random

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    dataset = MosquitoDataset(
        root_dir=cfg.DATA_DIR,
        split=split,
        transform=transform
    )

    print("Dataset size:", len(dataset))

    image, _ = dataset[random.randint(0, len(dataset)-1)]

    from collections import Counter

    labels = [label for _, label in dataset.samples]

    img = image.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":

    # For smaller dataset, in future use cfg.CLASS_NAMES
    # mosquitos = {
    #     "Aedes_Atlanticus": 0,
    #     "Aedes_Infirmatus": 1,
    #     "Orthopodomyia_Signifera": 2,
    #     "Psoraphora_Howardii": 3,
    #     "Psorophora_Ciliata": 4
    # }
    # for m in mosquitos:
    #     src = f'{cfg.DATA_DIR}/{m}'
    #     dst = f'{cfg.DATA_DIR}/processed/{m}'
    #     rename_and_move_imgs(src, dst, mosquitos[m])
    pass