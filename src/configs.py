# Configurations for Model

# ===============================
# DATA PATHS
# ===============================

DATA_DIR = "data/mosquito_data"
TRAIN_DIR = f"{DATA_DIR}/train"
TEST_DIR = f"{DATA_DIR}/test"
VAL_DIR = f"{DATA_DIR}/val"

UPLOAD_DIR = "data/uploaded"
PROCESSED_DIR = "data/processed"

# ====================================
# CHECKPOINTS (Saving Trained Model)
# ====================================

MN = "RN152_500" # Training model name

CHECKPOINT_DIR = f"checkpoints/{MN}" # Working path - Change for each trained model
MODEL_DIR = f"{CHECKPOINT_DIR}/BEST_model.pth"
FINAL_DIR = f'{CHECKPOINT_DIR}/FINAL_model.pth'

MODELS_DIR = f'{CHECKPOINT_DIR}/{MN}_'
# Logging
LOG_DIR = f'{CHECKPOINT_DIR}'
LOG_PATH = f'{LOG_DIR}/logs.csv'

# Classificaiton Report
CLASSIFICATION_REPORT_DIR = f'{CHECKPOINT_DIR}/{MN}_cr.txt'

# ===============================
# DEVICE
# ===============================

# Can check for GPU availability with utils.check_device()
DEVICE = "cuda" # Falls back to CPU if unavailable

# ===============================
# RANDOM SEED
# ===============================

SEED = 42

# ===============================
# CLASS NAMES - CHECK OVER ONCE DATA IS HERE
# ===============================

# NO MALES
# MOSQ_MAP = {
#     0: "Aedes aegypti",
#     1: "Aedes albopictus",
#     2: "Aedes atlanticus",
#     3: "Aedes atropalpus",
#     4: "Aedes canadensis",
#     5: "Aedes infirmatus",
#     6: "Aedes sollicitans",
#     7: "Aedes taeniorhynchus",
#     8: "Aedes triseriatus",
#     9: "Aedes vexans",
#     10: "Anopheles crucians",
#     11: "Anopheles punctipennis",
#     12: "Anopheles quadrimaculatus",
#     13: "Coquillettidia perturbans",
#     14: "Culex erraticus",
#     15: "Culex salinarius",
#     16: "Culex territans",
#     17: "Culex Nigripalpus",
#     18: "Culex coronator",
#     19: "Culex pipiens_restuans",
#     20: "Culiseta melanura",
#     21: "Orthopodomyia signifera",
#     22: "Psorophora ciliata",
#     23: "Psorophora columbiae",
#     24: "Psorophora ferox",
#     25: "Psoraphora howardii",
#     26: "Toxorhynchites rutilus",
#     27: "Uranotaenia sapphirina"
# }

# int: str (ALL)
MOSQ_MAP = {
    0: "Aedes aegypti",
    1: "Aedes albopictus",
    2: "Aedes atlanticus",
    3: "Aedes atropalpus",
    4: "Aedes canadensis",
    5: "Aedes infirmatus",
    6: "Aedes sollicitans",
    7: "Aedes taeniorhynchus",
    8: "Aedes triseriatus",
    9: "Aedes vexans",
    10: "Anopheles crucians",
    11: "Anopheles punctipennis",
    12: "Anopheles quadrimaculatus",
    13: "Coquillettidia perturbans",
    14: "Culex erraticus",
    15: "Culex salinarius",
    16: "Culex territans",
    17: "Culex Nigripalpus",
    18: "Culex coronator",
    19: "Culex pipiens_restuans",
    20: "Culiseta melanura",
    21: "Males",
    22: "Orthopodomyia signifera",
    23: "Psorophora ciliata",
    24: "Psorophora columbiae",
    25: "Psorophora ferox",
    26: "Psoraphora howardii",
    27: "Toxorhynchites rutilus",
    28: "Uranotaenia sapphirina"
}


# NO MALES
# IMG_MAP_REVERSED = {
#     "Aedes aegypti": 0,
#     "Aedes albopictus": 1,
#     "Aedes atlanticus": 2,
#     "Aedes atropalpus": 3,
#     "Aedes canadensis": 4,
#     "Aedes infirmatus": 5,
#     "Aedes sollicitans": 6,
#     "Aedes taeniorhynchus": 7,
#     "Aedes triseriatus": 8,
#     "Aedes vexans": 9,
#     "Anopheles crucians": 10,
#     "Anopheles punctipennis": 11,
#     "Anopheles quadrimaculatus": 12,
#     "Coquillettidia perturbans": 13,
#     "Culex erraticus": 14,
#     "Culex salinarius": 15,
#     "Culex territans": 16,
#     "Culex Nigripalpus": 17,
#     "Culex coronator": 18,
#     "Culex pipiens_restuans": 19,
#     "Culiseta melanura": 20,
#     "Orthopodomyia signifera": 21,
#     "Psorophora ciliata": 22,
#     "Psorophora columbiae": 23,
#     "Psorophora ferox": 24,
#     "Psoraphora howardii": 25,  # Should be Psorophora howardii
#     "Toxorhynchites rutilus": 26,
#     "Uranotaenia sapphirina": 27
# }

# str: int (ALL)
IMG_MAP_REVERSED = {
    "Aedes aegypti": 0,
    "Aedes albopictus": 1,
    "Aedes atlanticus": 2,
    "Aedes atropalpus": 3,
    "Aedes canadensis": 4,
    "Aedes infirmatus": 5,
    "Aedes sollicitans": 6,
    "Aedes taeniorhynchus": 7,
    "Aedes triseriatus": 8,
    "Aedes vexans": 9,
    "Anopheles crucians": 10,
    "Anopheles punctipennis": 11,
    "Anopheles quadrimaculatus": 12,
    "Coquillettidia perturbans": 13,
    "Culex erraticus": 14,
    "Culex salinarius": 15,
    "Culex territans": 16,
    "Culex Nigripalpus": 17,
    "Culex coronator": 18,
    "Culex pipiens_restuans": 19,
    "Culiseta melanura": 20,
    "Males": 21,
    "Orthopodomyia signifera": 22,
    "Psorophora ciliata": 23,
    "Psorophora columbiae": 24,
    "Psorophora ferox": 25,
    "Psoraphora howardii": 26,  # Should be Psorophora howardii
    "Toxorhynchites rutilus": 27,
    "Uranotaenia sapphirina": 28
}

# ===============================
# MODEL SETTINGS
# ===============================

MODEL_NAME = "resnet152" # Options: 18, 34, 50, 101, 152
NUM_CLASSES = 29 # 29 TOTAL (FULL DATASET) -- 28 w/o "Males" class

# ===============================
# TRAINING HYPERPARAMETERS
# ===============================

BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 500

SAVE_EPOCHS = 50 # Save every X epochs

# Can check for CPU count with utils.check_device()
NUM_WORKERS = 50

if __name__ == "__main__":
    import train as t

    t.main()