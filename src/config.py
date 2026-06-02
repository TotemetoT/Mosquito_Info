# Configurations for Model

# ===============================
# DATA PATHS
# ===============================

DATA_DIR = "data/mosquito_data"

# ===============================
# MODEL SETTINGS
# ===============================

MODEL_NAME = "resnet34"
NUM_CLASSES = 30

# ===============================
# TRAINING HYPERPARAMETERS
# ===============================

BATCH_SIZE = 30
LR = 1e-4
EPOCHS = 20

# ===============================
# CHECKPOINTS (Trained Model)
# ===============================

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "checkpoints/best_model.pth"
FINAL_MODEL_PATH = "checkpoints/final_model.pth"
LOG_PATH = "checkpoints/training_logs.pth"

# ===============================
# DEVICE
# ===============================

DEVICE = "cuda" # Falls back to CPU if unavailable

# ===============================
# RANDOM SEED
# ===============================

SEED = 42

# ===============================
# CLASS NAMES - CHECK OVER ONCE DATA IS HERE
# ===============================

CLASS_NAMES = {
    "Aedes aegypti",
    "Aedes albopictus",
    "Aedes atlanticus",
    "Aedes atropalpus",
    "Aedes canadensis",
    "Aedes infirmatus",
    "Aedes sollicitans",
    "Aedes taeniorhynchus",
    "Aedes triseriatus",
    "Aedes vexans",
    "Anopheles crucians",
    "Anopheles punctipennis",
    "Anopheles quadrimaculatus",
    "Coquillettidia perturbans",
    "Culex erraticus",
    "Culex pipiens/restuans",
    "Culex salinarius",
    "Culex territans",
    "Culiseta melanura",
    "Orthopodomyia signifera",
    "Psorophora ciliata",
    "Psorophora columbiae",
    "Psorophora ferox",
    "Psorophora howardii",
    "Uranotaenia sapphirina",
    "Culex Nigripalpus",
    "Culex coronator",
    "Culex pipiens_restuans",
    "Psoraphora howardii",
    "Toxorhynchites rutilus"
}