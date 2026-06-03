# Util functions

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

check_device()