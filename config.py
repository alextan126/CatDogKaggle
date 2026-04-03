from pathlib import Path
import torch

# Hardware and reproducibility
SEED = 42
CPU_THREADS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and Training Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
TUNER_EPOCHS = 6
FINAL_EPOCHS = 12
MAX_TRIALS = 60
EXECUTIONS_PER_TRIAL = 1

# Paths
DATASET_ROOT = Path.cwd() / 'kagglecatsanddogs_5340' / 'PetImages'
if not DATASET_ROOT.exists():
    # Fallback in case they are directly under the folder
    DATASET_ROOT = Path.cwd() / 'kagglecatsanddogs_5340'

ARTIFACTS_DIR = Path.cwd() / 'artifacts'
TUNER_DIR = ARTIFACTS_DIR / 'optuna_study'
MODEL_EXPORT_PATH = ARTIFACTS_DIR / 'best_resnet50_model.pth'

# Classes
CLASS_NAMES = ['Cat', 'Dog']