# Training methods for model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import *

from dataset import MosquitoDataset
from model import get_model
import configs as cfg

def train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1) # CHECK HERE
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    return epoch_loss, epoch_acc

def validate(
        model,
        dataloader,
        criterion,
        device
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1) # CHECK HERE
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)

    return val_loss, val_acc

def main():
    device = cfg.DEVICE

    # ===============================
    # IMAGE TRANSFORMATIONS
    # ===============================
    train_transform = Compose([
        Resize((300,300)),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = Compose([
        Resize((300,300)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ===============================
    # LOAD DATASETS & DATALOADERS
    # ===============================

    # TRAINING
    train_dataset = MosquitoDataset(
        root_dir=cfg.DATA_DIR,
        split="train",
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS
    )

    # VALIDATION
    val_dataset = MosquitoDataset(
        root_dir = cfg.DATA_DIR,
        split = "val",
        transform = val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS
    )

    # TESTING
    test_dataset = MosquitoDataset(
        root_dir = cfg.DATA_DIR,
        split = "test",
        transform = val_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS
    )

    # =================================
    # CREATE MODEL, LOSS, & OPTIMIZER
    # =================================
    model = get_model(
        num_classes=cfg.NUM_CLASSES,
        model_name=cfg.MODEL_NAME,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    # ===============================
    # TRAINING LOOP
    # ===============================
    best_val_acc = 0.0

    for epoch in range(cfg.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch+1}/{cfg.EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.BEST_MODEL_DIR)

    model.load_state_dict(
        torch.load(cfg.BEST_MODEL_DIR)
    )

    test_loss, test_acc = validate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"Test Loss: {test_loss:.4f}, "
        f"Test Acc: {test_acc:.4f}"
    )

if __name__ == "__main__":
    main()