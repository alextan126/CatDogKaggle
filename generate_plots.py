import json
import optuna
import torch
from pathlib import Path

from config import DATASET_ROOT, CLASS_NAMES, MODEL_EXPORT_PATH, DEVICE, ARTIFACTS_DIR
from data_pipeline import get_dataloaders
from model import build_model
from visualization import plot_roc_and_pr, show_prediction_panel, plot_parallel_coordinates

def generate_plots():
    print("=== Re-generating Evaluation Plots ===")
    
    # 1. Rebuild the dataset loaders
    print("Loading datasets...")
    _, val_loader, dataset_classes = get_dataloaders(DATASET_ROOT)
    
    # 2. Parse the training summary to get the best hyperparameters
    summary_path = ARTIFACTS_DIR / 'training_summary.json'
    if not summary_path.exists():
        print(f"Could not find {summary_path}. You need to extract the best hyperparameters manually.")
        # Fallback to the known best hyperparameters from the log
        best_hp = {'dense_units': 1024, 'dropout_rate': 0.4, 'learning_rate': 0.0001}
    else:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            best_hp = summary['best_hyperparameters']
            
    print(f"Using best hyperparameters: {best_hp}")
    
    # 3. Rebuild the model and load the saved weights
    print("Loading saved model weights...")
    best_model = build_model(best_hp['dense_units'], best_hp['dropout_rate']).to(DEVICE)
    
    if MODEL_EXPORT_PATH.exists():
        best_model.load_state_dict(torch.load(MODEL_EXPORT_PATH, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Could not find saved model weights at {MODEL_EXPORT_PATH}")
        
    best_model.eval()
    
    # 4. Generate the inference-based plots (with autocast fix applied)
    print("Generating ROC and PR curves...")
    plot_roc_and_pr(best_model, val_loader, dataset_classes)
    
    print("Generating 5x5 prediction panel...")
    show_prediction_panel(best_model, val_loader, dataset_classes)
    
    # 5. Reconstruct the Optuna Study from the log to generate the parallel coordinates plot
    print("Reconstructing Optuna study from log file...")
    study = optuna.create_study(direction="maximize")
    
    log_path = Path('training_output.log')
    if log_path.exists():
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if "Trial" in line and "finished with value" in line and "parameters:" in line:
                try:
                    # Parse lines like:
                    # [I 2026-04-03 05:19:34,180] Trial 32 finished with value: 0.990796318527411 and parameters: {'dense_units': 1024, 'dropout_rate': 0.4, 'learning_rate': 0.0001}. Best is trial 32...
                    value_str = line.split("finished with value: ")[1].split(" and parameters: ")[0]
                    params_str = line.split("parameters: ")[1].split(". Best is")[0].replace("'", '"')
                    
                    value = float(value_str)
                    params = json.loads(params_str)
                    
                    trial = optuna.trial.create_trial(
                        params=params,
                        distributions={
                            "dense_units": optuna.distributions.IntDistribution(256, 1024, step=256),
                            "dropout_rate": optuna.distributions.FloatDistribution(0.2, 0.6, step=0.1),
                            "learning_rate": optuna.distributions.CategoricalDistribution([1e-3, 1e-4, 1e-5])
                        },
                        value=value
                    )
                    study.add_trial(trial)
                except Exception as e:
                    print(f"Failed to parse line: {line.strip()} - {e}")
                    
        print(f"Reconstructed {len(study.trials)} trials.")
        if len(study.trials) > 0:
            print("Generating Parallel Coordinates plot...")
            plot_parallel_coordinates(study)
    else:
        print("Could not find training_output.log to reconstruct the Optuna study.")
        
    print("Done! All plots have been generated in the artifacts/ directory.")

if __name__ == "__main__":
    generate_plots()
