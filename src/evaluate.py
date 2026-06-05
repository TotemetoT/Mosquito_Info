# Evaluate using Macro F1 Score & training vs. validation graphs & confusion matrix

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from torch.utils.data import DataLoader

from dataset import MosquitoDataset
from model import get_model
import configs as cfg

device = cfg.DEVICE

model = get_model(
    num_classes=cfg.NUM_CLASSES,
    model_name=cfg.MODEL_NAME
).to(device)

model.load_state_dict(
    torch.load(
        cfg.BEST_MODEL_DIR,
        map_location=device
    )
)

model.eval()

from torchvision.transforms import *

test_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = MosquitoDataset(
    root_dir=cfg.DATA_DIR,
    split="test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False
)

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

class_names = [
    cfg.MOSQ_MAP[i]
    for i in sorted(cfg.MOSQ_MAP.keys())
]

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(10, 10)
)

disp.plot(
    ax=ax,
    xticks_rotation=90
)

plt.tight_layout()

plt.savefig(
    cfg.CONFUSION_MATRIX_DIR,
    dpi=300,
    bbox_inches="tight"
)