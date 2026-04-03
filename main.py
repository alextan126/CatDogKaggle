import json
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import (
    SEED, CPU_THREADS, DATASET_ROOT, CLASS_NAMES, ARTIFACTS_DIR, TUNER_DIR,
    MODEL_EXPORT_PATH, TUNER_EPOCHS, FINAL_EPOCHS, MAX_TRIALS, DEVICE
)
from data_pipeline import sanitize_dataset, get_dataloaders
from model import build_model
from visualization import (
    plot_training_curves, plot_roc_and_pr, plot_parallel_coordinates, show_prediction_panel
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

def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

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

def objective(trial, train_loader, val_loader):
    # Hyperparameters to tune
    dense_units = trial.suggest_int('dense_units', 256, 1024, step=256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1)
    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 1e-4, 1e-5])
    
    model = build_model(dense_units, dropout_rate).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    best_val_acc = 0.0
    
    for epoch in range(TUNER_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        
        trial.report(val_acc, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    return best_val_acc

def main():
    print("=== Step 1: Environment Setup ===")
    setup_environment()

    print("\n=== Step 2: Data Sanitization ===")
    sanitization_summary = sanitize_dataset(DATASET_ROOT, CLASS_NAMES)
    print("Sanitization Summary:")
    print(sanitization_summary)

    print("\n=== Step 3: Building Datasets ===")
    train_loader, val_loader, dataset_classes = get_dataloaders(DATASET_ROOT)
    print(f"Classes found: {dataset_classes}")

    print("\n=== Step 4: Hyperparameter Tuning ===")
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader), 
        n_trials=MAX_TRIALS
    )

    best_hp = study.best_params
    print(f"\nBest hyperparameters found: {best_hp}")

    print("\n=== Step 5: Final Model Training ===")
    best_model = build_model(best_hp['dense_units'], best_hp['dropout_rate']).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_hp['learning_rate'])
    scaler = GradScaler()
    
    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': []
    }
    
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0
    
    for epoch in range(FINAL_EPOCHS):
        print(f"Epoch {epoch+1}/{FINAL_EPOCHS}")
        train_loss, train_acc = train_one_epoch(best_model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate_one_epoch(best_model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(best_model.state_dict(), MODEL_EXPORT_PATH)
            print(f"Saved new best model to {MODEL_EXPORT_PATH}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best weights for evaluation
    best_model.load_state_dict(torch.load(MODEL_EXPORT_PATH))

    print("\n=== Step 6: Evaluation & Visualization ===")
    plot_training_curves(history)
    plot_roc_and_pr(best_model, val_loader, dataset_classes)
    plot_parallel_coordinates(study)
    show_prediction_panel(best_model, val_loader, dataset_classes)

    print("\n=== Step 7: Saving Summary ===")
    summary = {
        'best_hyperparameters': best_hp,
        'best_val_accuracy': float(max(history['val_accuracy'])),
        'best_val_loss': float(min(history['val_loss'])),
        'model_export_path': str(MODEL_EXPORT_PATH),
    }
    summary_path = ARTIFACTS_DIR / 'training_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved training summary to {summary_path}")
    print("Pipeline Complete!")

if __name__ == "__main__":
    main()
