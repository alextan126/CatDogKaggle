# High-Performance Image Classification Pipeline

This repository contains a high-performance PyTorch pipeline for binary image classification (Cats vs. Dogs). It is specifically optimized for execution on NVIDIA DGX Spark hardware (Grace Blackwell / GB10 architecture).

## Overview

The pipeline leverages a pre-trained `ResNet-50` backbone with a custom, tunable classification head. It uses `Optuna` to perform a hyperparameter search across learning rates, dropout rates, and the dimensionality of the latent space (dense units).

### Key Features
- **Robust Data Sanitization**: Automatically scans the dataset to remove corrupted 0KB files, the specific "666" file, and repairs JPEGs with missing EOF markers using PIL.
- **Mixed Precision Training**: Utilizes `torch.cuda.amp.autocast` to maximize throughput on the Blackwell GPU.
- **Optuna Hyperparameter Tuning**: Uses a `MedianPruner` to efficiently search for the optimal architecture and regularization balance.
- **Automated Evaluation**: Generates training curves, ROC/Precision-Recall curves, a 5x5 visual prediction panel, and an interactive parallel coordinates plot.

## Project Structure
```text
.
├── config.py                # Hyperparameters, paths, and hardware config
├── data_pipeline.py         # Dataset sanitization and DataLoader creation
├── download_dataset.py      # Helper script to download the Microsoft dataset
├── generate_plots.py        # Helper script to regenerate plots from saved weights
├── main.py                  # The main training and tuning loop
├── model.py                 # ResNet-50 architecture and custom tunable head
├── requirements.txt         # Python dependencies
├── run_docker_training.sh   # Bash script to run the pipeline in an NVIDIA container
└── visualization.py         # Matplotlib and Plotly evaluation functions
```

## Running the Pipeline

To bypass TensorFlow/Keras deprecation issues on the GB10 architecture, this pipeline is designed to run inside the official NVIDIA PyTorch Docker container.

1. **Download the Dataset**:
   ```bash
   python download_dataset.py
   ```

2. **Execute the Training Script**:
   The provided bash script will automatically pull the correct container, mount your workspace, install the requirements, and execute `main.py`.
   ```bash
   ./run_docker_training.sh | tee training_output.log
   ```

## Results and Visualizations

All training artifacts, including the best model weights (`best_resnet50_model.pth`) and the evaluation plots, are saved in the `artifacts/` directory.

### Artifacts Interpretation & Training Dynamics

#### Hyperparameter Grid Search & Final Model Selection
The training process was divided into two distinct phases:
1. **Grid Search (Optuna Study):** The pipeline first ran a hyperparameter search across multiple trials. To quickly evaluate each combination, every trial was trained for only **6 epochs**. A MedianPruner was used to stop unpromising trials early.
2. **Final Training:** After the study concluded, the best hyperparameters were selected:
   * **Dense Units (Latent Space):** `1024`
   * **Dropout Rate:** `0.4`
   * **Learning Rate:** `0.0001` (1e-4)

The model was then rebuilt from scratch using these optimal parameters and trained for a maximum of **12 epochs** to achieve peak performance.

#### Why Early Stopping?
During the final 12-epoch training phase, the training automatically halted after Epoch 7, with the final model weights being restored from **Epoch 6**. 

This is due to an **Early Stopping** mechanism monitoring the Validation Loss:
* **Epoch 6:** Validation Loss reached its absolute minimum (`0.0290`) with a Validation Accuracy of `98.88%`.
* **Epoch 7:** Validation Loss increased to `0.0313`, and accuracy slightly dipped.

In deep learning, when training loss continues to decrease but validation loss starts to increase, the model is beginning to **overfit** (memorizing the training data rather than learning generalizable features). The early stopping callback detected this divergence at Epoch 7, halted the training to save compute resources, and optimally selected the weights from Epoch 6, representing the model at its peak generalization capability.

#### Graph Interpretations

**1. Training Curves (`training_curves.png`)**
* **Loss Curve:** Shows the training loss steadily decreasing while the validation loss drops and then plateaus. The slight uptick in validation loss at the very end visually confirms the onset of overfitting, justifying the early stop at Epoch 6.
* **Accuracy Curve:** Demonstrates rapid learning in the first few epochs, with both training and validation accuracy converging and stabilizing near ~99%, indicating a highly capable and stable model.

**2. ROC and Precision-Recall Curves (`roc_pr_curves.png`)**
* **ROC Curve (Receiver Operating Characteristic):** Plots the True Positive Rate (Sensitivity/Recall) against the False Positive Rate (1 - Specificity) across various probability thresholds. The curve sharply hugs the top-left corner, resulting in an Area Under the Curve (AUC) very close to 1.0. This means the model has a near-perfect ability to distinguish between the two classes (Cats vs. Dogs) across all classification thresholds. A random guess would be a diagonal line (AUC = 0.5).
* **PR Curve (Precision-Recall):** Plots Precision (Positive Predictive Value) against Recall (True Positive Rate). The curve stays tightly packed in the top-right corner, indicating that the model maintains extremely high precision (few false positives, i.e., rarely calling a dog a cat) even when recall is high (finding almost all true positives, i.e., successfully identifying all cats). This is especially useful for confirming model robustness.

**3. Parallel Coordinates Plot (`parallel_coordinates.html`)**
This interactive Optuna plot visualizes the hyperparameter grid search. Each line represents a single trial connecting different hyperparameter choices.
* **Interpretation:** By tracing the lines that reach the highest points on the `Objective Value` (Validation Accuracy) axis on the far right, you can visually identify the optimal "path" of hyperparameters. It reveals how different combinations of `learning_rate`, `dropout_rate`, and `dense_units` interact. For example, it might show that a lower learning rate requires less dropout to achieve peak performance, or that a larger latent space (more dense units) is only beneficial with higher regularization. *(Open `artifacts/parallel_coordinates.html` in a web browser to view and interact with the Plotly chart).*

**4. Visual Validation Panel (`prediction_panel.png`)**
A 5x5 grid displaying 25 random images from the validation set. Each image is overlaid with the model's predicted label and the true ground-truth label. This serves as a qualitative "sanity check" to ensure the model isn't just mathematically accurate, but is genuinely recognizing the visual features of cats and dogs correctly.
