# Boob Bam Bop

import os
import torch
from PIL import Image
from torch.utils.data import Dataset

import configs as cfg
import utils as u

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

class MosquitoDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):

        self.root_dir = root_dir
        self.split = split
        self.train_dir = os.path.join(root_dir, "train")
        self.val_dir = os.path.join(root_dir, "val")
        self.test_dir = os.path.join(root_dir, "test")

        self.transform = transform

        self.map = cfg.MOSQ_MAP

        self.samples = []

        self._load_samples()
    
    def _load_samples(self):
            for filename in os.listdir(os.path.join(self.root_dir, self.split)):
                if not filename.lower().endswith(tuple(VALID_EXTENSIONS)):
                    continue
                try:
                    class_code = u.identify_img(filename) # (Bool, Class Name)
                except ValueError:
                    continue
                if class_code[0]:
                    img_path = os.path.join(self.root_dir, self.split, filename)
                    self.samples.append((img_path, int(filename[:2])))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    def get_class_name(self, label):
        return u.identify_img(label)[1]
    

