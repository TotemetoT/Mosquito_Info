from pathlib import Path
import shutil
import random

import configs as cfg

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

def separate_data(dir, out):
    dir = Path(dir)
    out = Path(out)

    out.mkdir(parents=True, exist_ok=True)

    for file in dir.iterdir():
        if file.is_file():
            # Extract the last three digits of the filename
            # Example: 01007.jpg -> 007 -> 7
            image_num = int(file.stem[-3:])

            # Move every 7th image to the output directory
            if int(image_num) % 7 == 0:
                print(f"Moving {file.name} to test set")
                shutil.move(str(file), str(out / file.name))

def move_files(dir,out):
    dir = Path(dir)
    out = Path(out)

    out.mkdir(parents=True, exist_ok=True)

    for file in dir.iterdir():
        if file.is_file():
            shutil.move(str(file), str(out / file.name))

def split_dataset(dataset_dir, output_dir, seed=42):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    random.seed(cfg.SEED)

    # Create split folders
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for folder in [train_dir, val_dir, test_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # Loop through species folders
    for species_dir in dataset_dir.iterdir():
        if not species_dir.is_dir():
            continue

        original_dir = species_dir / "original"
        augmented_dir = species_dir / "augmented"

        originals = list(original_dir.glob("*"))
        random.shuffle(originals)

        n = len(originals)

        train_end = int(0.70 * n)
        val_end = train_end + int(0.15 * n)

        train = originals[:train_end]
        val = originals[train_end:val_end]
        test = originals[val_end:]

        # Copy training originals + augmentations
        for img in train:
            shutil.copy2(img, train_dir / img.name)

            image_num = int(img.stem[-3:])

            # Find the 6 augmentations following the original
            for i in range(1, 7):
                aug_num = image_num + i
                aug_name = f"{img.stem[:-3]}{aug_num:03d}{img.suffix}"
                aug_path = augmented_dir / aug_name

                if aug_path.exists():
                    shutil.copy2(
                        aug_path,
                        train_dir / aug_name
                    )

        # Copy validation originals only
        for img in val:
            shutil.copy2(img, val_dir / img.name)

        # Copy test originals only
        for img in test:
            shutil.copy2(img, test_dir / img.name)

        print(
            f"{species_dir.name}: "
            f"{len(train)} train originals, "
            f"{len(val)} validation originals, "
            f"{len(test)} test originals"
        )

if __name__ == "__main__":
    
    split_dataset("data/processed", "data/split")