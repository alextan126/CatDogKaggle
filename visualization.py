import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization as vis
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from config import ARTIFACTS_DIR, DEVICE

def plot_training_curves(history: dict) -> None:
    """Plot and save training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='train_loss')
    axes[0].plot(history['val_loss'], label='val_loss')
    axes[0].set_title('Loss vs Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history['train_accuracy'], label='train_accuracy')
    axes[1].plot(history['val_accuracy'], label='val_accuracy')
    axes[1].set_title('Accuracy vs Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    out_path = ARTIFACTS_DIR / 'training_curves.png'
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved training curves to {out_path}")

def collect_labels_and_scores(model: nn.Module, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Run inference to collect true labels and predicted probabilities."""
    model.eval()
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            
            all_labels.extend(labels.numpy())
            all_scores.extend(probs.cpu().numpy())
            
    return np.array(all_labels), np.array(all_scores)

def cat_binary_targets(y_true: np.ndarray, y_score_dog: np.ndarray, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Convert labels and scores to focus on the 'Cat' class."""
    cat_index = class_names.index('Cat')
    y_true_cat = (y_true == cat_index).astype(np.int32)
    if cat_index == 0:
        y_score_cat = 1.0 - y_score_dog
    else:
        y_score_cat = y_score_dog
    return y_true_cat, y_score_cat

def plot_roc_and_pr(model: nn.Module, dataloader: DataLoader, class_names: list[str]) -> None:
    """Plot and save ROC and Precision-Recall curves for the Cat class."""
    y_true, y_score_dog = collect_labels_and_scores(model, dataloader)
    y_true_cat, y_score_cat = cat_binary_targets(y_true, y_score_dog, class_names)

    fpr, tpr, _ = roc_curve(y_true_cat, y_score_cat)
    precision, recall, _ = precision_recall_curve(y_true_cat, y_score_cat)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0].set_title('ROC Curve for Cats')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right')

    axes[1].plot(recall, precision, label=f'AUC = {pr_auc:.4f}')
    axes[1].set_title('Precision-Recall Curve for Cats')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='lower left')

    plt.tight_layout()
    out_path = ARTIFACTS_DIR / 'roc_pr_curves.png'
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved ROC and PR curves to {out_path}")

def plot_parallel_coordinates(study: optuna.Study) -> None:
    """Save an interactive HTML parallel coordinates plot of Optuna trials."""
    fig = vis.plot_parallel_coordinate(study)
    html_path = ARTIFACTS_DIR / 'parallel_coordinates.html'
    fig.write_html(str(html_path))
    print(f"Saved parallel coordinates plot to: {html_path}")

def decode_binary_prediction(score_dog: float, class_names: list[str]) -> str:
    predicted_index = int(score_dog >= 0.5)
    return class_names[predicted_index].lower()

def show_prediction_panel(model: nn.Module, dataloader: DataLoader, class_names: list[str], n_rows: int = 5, n_cols: int = 5) -> None:
    """Save a 5x5 grid of validation images with true and predicted labels."""
    model.eval()
    total_images = n_rows * n_cols
    
    # Get a single batch of images
    images, labels = next(iter(dataloader))
    images = images[:total_images]
    labels = labels[:total_images]
    
    with torch.no_grad():
        outputs = model(images.to(DEVICE)).squeeze(1)
        probs = torch.sigmoid(outputs).cpu().numpy()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    axes = axes.flatten()

    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, ax in enumerate(axes):
        if i < len(images):
            image = images[i]
            label = labels[i].item()
            score_dog = probs[i]
            
            # Denormalize and convert to numpy HWC
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            img_np = image.permute(1, 2, 0).numpy()

            true_label = class_names[label].lower()
            pred_label = decode_binary_prediction(score_dog, class_names)

            ax.imshow(img_np)
            ax.set_title(f'label: {true_label}, pred: {pred_label}', fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    out_path = ARTIFACTS_DIR / 'prediction_panel.png'
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved prediction panel to {out_path}")
