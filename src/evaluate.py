# Evaluate using Macro F1 Score & training vs. validation graphs & confusion matrix

import torch
import matplotlib.pyplot as plt
from torchvision.transforms import *
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import MosquitoDataset
from model import get_model
import configs as cfg

# =====================
# LOAD TRAINED MODEL
# =====================

def load_model(m):
    device = cfg.DEVICE

    model = get_model(
        num_classes=cfg.NUM_CLASSES,
        model_name=cfg.MODEL_NAME
    ).to(device)

    model.load_state_dict(
        torch.load(
            m,
            map_location=device
        )
    )

    model.eval()

    return model

# ==========================
# TEST DATASET DATALOADER
# ==========================

def test_loader():
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

    return DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False
    )

# ========================
# CLASSIFICATION REPORT
# ========================

def cr(y_true, y_pred):
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

    return class_names

# =======================
# EVALUATE MODEL
# =======================

def evaluate_model(model):
    device = cfg.DEVICE

    y_true = []
    y_pred = []

    dataloader = test_loader()

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                dim=1
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    return y_true, y_pred

# ===================
# CONFUSION MATRIX
# ===================

def cm(
    y_true,
    y_pred,
    class_names,
    save_path
):
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
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

# ==========================
# MAIN EVALUATE FUNCTION
# ==========================

def evaluate(m):
    model = load_model(m)

    y_true, y_pred = evaluate_model(
        model,
    )

    class_names = cr(
        y_true,
        y_pred
    )

    cm(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=cfg.CONFUSION_MATRIX_DIR
    )

# =========================
# Loss Plot
# =========================

def plot_loss():
    df = pd.read_csv(cfg.LOG_PATH)
    epochs = df["epoch"]

    plt.figure()
    plt.plot(epochs, df["train_loss"], label="Train Loss")
    plt.plot(epochs, df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f'{cfg.LOG_DIR}/Loss_plot')

# =========================
# Accuracy Plot
# =========================

def plot_acc():
    df = pd.read_csv(cfg.LOG_PATH)
    epochs = df["epoch"]
    plt.figure()
    plt.plot(epochs, df["train_acc"], label="Train Accuracy")
    plt.plot(epochs, df["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f'{cfg.LOG_DIR}/Accuracy_plot')

if __name__ == "__main__":
    best = cfg.MODEL_DIR
    final = cfg.FINAL_DIR
    evaluate(best)