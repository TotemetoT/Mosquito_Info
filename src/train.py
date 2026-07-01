# Training methods for model

import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import *
from tqdm import tqdm

import evaluate as eval
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

    max_batch_loss = 0.0
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="training", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        max_batch_loss = max(max_batch_loss, loss.item())

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1) # CHECK HERE
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    return epoch_loss, epoch_acc, max_batch_loss

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
        
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
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

def main(c):
    device = cfg.DEVICE

    # =================================
    # LOGGING SETUP
    # =================================

    csv_path = cfg.LOG_PATH

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "LR",
            "batch_loss",
            "batch_size"
        ])

    # ===============================
    # IMAGE TRANSFORMATIONS
    # ===============================
    train_transform = Compose([
        Resize((300,300)),
        RandomCrop(224),        # Comment if needed
        # RandomHorizontalFlip(), # Comment if needed
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = Compose([
        Resize((224, 224)),
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
        batch_size=c.batch_size, 
        shuffle=True,
        num_workers=c.num_workers
    )

    # VALIDATION
    val_dataset = MosquitoDataset(
        root_dir = cfg.DATA_DIR,
        split = "val",
        transform = val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=c.batch_size, 
        shuffle=False,
        num_workers=c.num_workers
    )

    # TESTING
    test_dataset = MosquitoDataset(
        root_dir = cfg.DATA_DIR,
        split = "test",
        transform = val_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=c.batch_size, 
        shuffle=False,
        num_workers=c.num_workers
    )

    # =================================
    # CREATE MODEL, LOSS, & OPTIMIZER
    # =================================
    model = get_model(
        num_classes=cfg.NUM_CLASSES,
        model_name=c.model_name,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=c.lr) 

    # ===============================
    # TRAINING LOOP
    # ===============================
    best_val_acc = 0.0

    for epoch in range(c.epochs): 
        train_loss, train_acc, max_batch_loss = train_one_epoch(
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
            f"Epoch [{epoch+1}/{c.epochs}] " 
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                optimizer.param_groups[0]['lr'],
                max_batch_loss,
                c.batch_size
            ])

        eval.plot_acc()
        eval.plot_loss()
        
        # Save every X epochs
        if (epoch + 1) % c.save_epochs == 0:
            torch.save(model.state_dict(), f"{cfg.CHECKPOINT_DIR}/{cfg.MN}_{epoch+1}.pth")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        #     torch.save(model.state_dict(), cfg.MODEL_DIR)

    model.load_state_dict(
        torch.load(f"{cfg.CHECKPOINT_DIR}/{cfg.MN}_{epoch+1}.pth", weights_only=True)
    )

    eval.evaluate(f"{cfg.CHECKPOINT_DIR}/{cfg.MN}_{epoch+1}.pth", f"Model: val acc = {val_acc}", c)

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

    return best_val_acc

if __name__ == "__main__":
    from configs import config as c
    main(c)