from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from config import (
    SEED, CPU_THREADS, DATASET_ROOT, ARTIFACTS_DIR, TUNER_DIR,
    MODEL_EXPORT_PATH, DEVICE, IMAGE_SIZE, BATCH_SIZE
)
from data_pipeline import sanitize_dataset
from model import build_model
from visualization import (
    plot_roc_and_pr, show_prediction_panel
)

def setup_environment():
    """Configure PyTorch seeds and directories."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TUNER_DIR.mkdir(parents=True, exist_ok=True)

    print(f'PyTorch version: {torch.__version__}')
    print(f'Device: {DEVICE}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Logical CPU threads requested: {CPU_THREADS}')

def get_dataset_class_dirs(dataset_root: Path) -> list[str]:
    """Return the class directory names present under the dataset root."""
    class_dirs = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders were found under {dataset_root}. "
            "Expected a layout like sample_data/cats and sample_data/dogs."
        )
    return class_dirs


def infer_model_head_params(checkpoint_path: Path) -> tuple[int, float]:
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    dense_units = state_dict['fc.0.weight'].shape[0]
    dropout_rate = 0.4
    return dense_units, dropout_rate

def get_eval_dataloader(dataset_root: Path):
    """Load all images from the dataset root into a single evaluation loader."""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=str(dataset_root), transform=transform)

    eval_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=CPU_THREADS,
        pin_memory=True
    )

    return eval_loader, full_dataset.classes

def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()
            
            with autocast():
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    print("=== Step 1: Environment Setup ===")
    setup_environment()

    print("\n=== Step 2: Data Sanitization ===")
    dataset_class_dirs = get_dataset_class_dirs(DATASET_ROOT)
    sanitization_summary = sanitize_dataset(DATASET_ROOT, dataset_class_dirs)
    print("Sanitization Summary:")
    print(sanitization_summary)

    print("\n=== Step 3: Building Datasets ===")
    val_loader, dataset_classes = get_eval_dataloader(DATASET_ROOT)
    print(f"Dataset folders found: {dataset_classes}")

    print("\n=== Step 4: Loading Saved Model ===")
    dense_units, dropout_rate = infer_model_head_params(MODEL_EXPORT_PATH)
    best_model = build_model(dense_units, dropout_rate).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    best_model.load_state_dict(torch.load(MODEL_EXPORT_PATH, map_location=DEVICE))
    best_model.eval()

    print("\n=== Step 5: One Epoch on the validation set ===")
    print("One epoch to get validation loss")
    val_loss, val_acc = validate_one_epoch(best_model, val_loader, criterion)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    print("\n=== Step 6: Evaluation & Visualization ===")
    plot_roc_and_pr(best_model, val_loader, dataset_classes)
    show_prediction_panel(best_model, val_loader, dataset_classes)

    print("Pipeline Complete!")

if __name__ == "__main__":
    main()
