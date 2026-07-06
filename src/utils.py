# Util functions & Proprocessing

import os
import torch
import random
from pathlib import Path
import shutil

from model import get_model
import configs as cfg
from configs import config as c

# ==============================
# CHECK FOR DEVICE (GPU/CPU)
# ==============================

def check_device():
    # Check for NVIDIA GPU, Apple Silicon, or default to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_cores = os.cpu_count()

    print(f"Using device: {device}  |  CPU Count: {num_cores}")

# ==============================
# PREPROCESSING METHODS
# ==============================

def rename_and_move_imgs(src_dir, dst_dir, fname, prefix="img"):
    """
    Takes a given directory of images and renames them for consistency

    @param src_dir: source directory
    @param dst_dir: destination directory
    @param fname: mosquito species
    """

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

def split_data(split, dir):
    """
    Goes through processed directories and sorts the data
    @param split: how to split data [train,val,test]
    """
    if sum(split) != 1:
        raise ValueError("Split doesn't add up to 1")
    
    train, val, test = split
    dir = Path(dir)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted([f for f in dir.iterdir() if f.suffix.lower() in img_exts])
    random.shuffle(files)
    num_files = len(files)

    # ===========================
    # CALCULATE SPLITS
    # ===========================

    train_files = train * num_files
    print(train_files)
    if train_files % 1 != 0:
        train_files = int(train_files) + 1

    test_files = test * num_files
    print(test_files)
    if test_files % 1 != 0 and train_files % 1 >= 0.5:
        test_files = int(test_files) + 1
    else: 
        test_files = int(test_files)
    
    val_files = num_files - (test_files + train_files)

    print(f'splits: train: {train_files} | val: {val_files} | test: {test_files} | {train_files + val_files + test_files} = {num_files}')
    
    # ==============================
    # MOVE DATA (TRAIN,VAL,TEST)
    # ==============================

    train_dir = Path(cfg.TRAIN_DIR)
    val_dir   = Path(cfg.VAL_DIR)
    test_dir  = Path(cfg.TEST_DIR)

    for f in files:
        if train_files != 0:
            shutil.copy2(f, train_dir)
            train_files -= 1
            print(f'Moving {f} to {train_dir} | {train_files}')
        elif test_files != 0:
            shutil.copy2(f, test_dir)
            test_files -= 1
            print(f'Moving {f} to {test_dir} | {test_files}')
        else:
            shutil.copy2(f, val_dir)
            val_files -= 1
            print(f'Moving {f} to {val_dir} | {val_files}')

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
    Tests the MosquitoDataset class by loading a sample split
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


    labels = [label for _, label in dataset.samples]

    label_dict = {}
    for label in labels:
        if label in label_dict:
            label_dict[label] += 1
        else:
            label_dict[label] = 1
    for l in label_dict:
        print(l, ":",label_dict[l])
    print(len(labels))

def vectorize(split, savefile):
    from dataset import MosquitoDataset
    from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor, Normalize
    from torch.utils.data import DataLoader

    # ===============================
    # IMAGE TRANSFORMATIONS
    # ===============================

    if split == "train":
        transform = Compose([
            Resize((300,300)),
            RandomCrop(224),        # Comment if needed
            # RandomHorizontalFlip(), # Comment if needed
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # ==============================
    # Load & Vectorize Dataset
    # ==============================

    dataset = MosquitoDataset(
        root_dir=cfg.DATA_DIR,
        split=split,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=c.batch_size, 
        shuffle=True,
        num_workers=c.num_workers
        )
    
    imgs, labels = [], []

    for img, label in dataloader:
        imgs.append(img)
        labels.append(label)

    torch.save({
        "images": torch.cat(imgs),
        "labels": torch.cat(labels)
    }, savefile)

# ==============================
# FIND BEST EPOCH
# ==============================

def best_epoch(file):
    """
    Finds the epoch that was saved as the best model from training

    @param file: string, logs.csv path
    @return tuple: (int, float)
    """
    import csv
    with open(file, "r") as f:
        reader = csv.reader(f)
        next(reader)

        best_val, best_epoch = 0.0, 0
        for line in reader:
            epoch, _, _, _, val, _, _, _ = line
            if float(val) >= best_val:
                best_val = float(val)
                best_epoch = int(epoch)
        
        return (best_epoch, best_val)
    

if __name__ == "__main__":

    mosquitos = cfg.IMG_MAP_REVERSED

    # for m in mosquitos:
    #     src = f'{cfg.UPLOAD_DIR}/{m}'
    #     dst = f'{cfg.PROCESSED_DIR}/{m}'
    #     rename_and_move_imgs(src, dst, mosquitos[m])

    # for m in mosquitos:
    #     print(f'Working on: {m}')
    #     src = f'{cfg.PROCESSED_DIR}/{m}'
    #     split = [.70,.15,.15]
    #     split_data(split, src)

    # for split in ["train", "val", "test"]:
    #     print(split)
    #     savefile = Path(f"data/vectorized/{split}.pth")
    #     vectorize(split, savefile)

    print(best_epoch(cfg.LOG_PATH))
