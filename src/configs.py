# Configurations for Model

# ===============================
# DATA PATHS
# ===============================

DATA_DIR = "data/mosquito_data"
TRAIN_DIR = f"{DATA_DIR}/train"
TEST_DIR = f"{DATA_DIR}/test"
VAL_DIR = f"{DATA_DIR}/val"

# ===============================
# MODEL SETTINGS
# ===============================

MODEL_NAME = "resnet34" # Options: 18, 34, 50
NUM_CLASSES = 5

# ===============================
# TRAINING HYPERPARAMETERS
# ===============================

BATCH_SIZE = 30
LR = 1e-4
EPOCHS = 20

NUM_WORKERS = 4

# ===============================
# CHECKPOINTS (Trained Model)
# ===============================

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "checkpoints/best_model.pth"
FINAL_MODEL_PATH = "checkpoints/final_model.pth"
CONFUSION_MATRIX_DIR = f"{CHECKPOINT_DIR}/cm.png"
LOG_PATH = "checkpoints/training_logs.pth"

# ===============================
# DEVICE
# ===============================

# Can check for GPU availability at utils.check_device()
DEVICE = "cuda" # Falls back to CPU if unavailable

# ===============================
# RANDOM SEED
# ===============================

SEED = 42

# ===============================
# CLASS NAMES - CHECK OVER ONCE DATA IS HERE
# ===============================

# SMALLER DATASET - SORTING IMAGE NAMES
IMG_MAP = {
    "Aedes_Atlanticus": 0,
    "Aedes_Infirmatus": 1,
    "Orthopodomyia_Signifera": 2,
    "Psoraphora_Howardii": 3,
    "Psorophora_Ciliata": 4
}

# SMALLER DATASET - IDENTIFYING SPECIES
MOSQ_MAP = {
    0: "Aedes_Atlanticus",
    1: "Aedes_Infirmatus",
    2: "Orthopodomyia_Signifera",
    3: "Psoraphora_Howardii",
    4: "Psorophora_Ciliata"
}


# FULL DATASET
# CLASS_NAMES = {
#     "Aedes aegypti",
#     "Aedes albopictus",
#     "Aedes atlanticus",
#     "Aedes atropalpus",
#     "Aedes canadensis",
#     "Aedes infirmatus",
#     "Aedes sollicitans",
#     "Aedes taeniorhynchus",
#     "Aedes triseriatus",
#     "Aedes vexans",
#     "Anopheles crucians",
#     "Anopheles punctipennis",
#     "Anopheles quadrimaculatus",
#     "Coquillettidia perturbans",
#     "Culex erraticus",
#     "Culex pipiens/restuans",
#     "Culex salinarius",
#     "Culex territans",
#     "Culex Nigripalpus",
#     "Culex coronator", 
#     "Culex pipiens_restuans",
#     "Culiseta melanura",
#     "Orthopodomyia signifera",
#     "Psorophora ciliata",
#     "Psorophora columbiae",
#     "Psorophora ferox",
#     "Psorophora howardii",
#     "Psoraphora howardii",
#     "Toxorhynchites rutilus",
#     "Uranotaenia sapphirina"
# }