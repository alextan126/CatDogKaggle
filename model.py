import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_model(dense_units: int, dropout_rate: float) -> nn.Module:
    """Builds a ResNet50 model with a tunable custom head for PyTorch."""
    # Load pre-trained ResNet50
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    
    # Freeze the base model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Get the number of features from the original fully connected layer
    in_features = model.fc.in_features
    
    # Replace the fully connected layer with a custom tunable head
    model.fc = nn.Sequential(
        nn.Linear(in_features, dense_units),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(dense_units, 1) # Binary classification output
    )
    
    return model
