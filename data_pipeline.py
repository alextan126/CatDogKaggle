import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import CPU_THREADS, IMAGE_SIZE, BATCH_SIZE, SEED, CLASS_NAMES

def require_dataset_root(dataset_root: Path, class_names: list[str]) -> None:
    """Ensure the dataset directories exist before proceeding."""
    missing_dirs = [class_name for class_name in class_names if not (dataset_root / class_name).exists()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Expected extracted dataset folders were not found. Missing: {missing_dirs}. "
            f"Expected layout: {dataset_root}/Cat and {dataset_root}/Dog."
        )

def sanitize_file(file_path: Path) -> dict[str, int]:
    """Check a single file and delete it if it's corrupted, named '666', or unreadable."""
    result = {
        'total_files': 1,
        'deleted_zero_kb': 0,
        'deleted_666': 0,
        'deleted_unreadable': 0,
        'retained_files': 0,
    }

    if file_path.stem == '666':
        file_path.unlink(missing_ok=True)
        result['deleted_666'] = 1
        return result
        
    if file_path.stat().st_size == 0:
        file_path.unlink(missing_ok=True)
        result['deleted_zero_kb'] = 1
        return result

    # Try to open the image to ensure it's valid and has 3 channels
    try:
        # 1. PIL verification
        with Image.open(file_path) as img:
            img.verify() # Verify it's actually an image
            
        # 3. Check for corrupt JPEG data (extraneous bytes before EOF)
        if file_path.suffix.lower() in ['.jpg', '.jpeg']:
            with open(file_path, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':
                    # Instead of deleting, we can try to fix it by saving a clean copy using PIL
                    try:
                        with Image.open(file_path) as img:
                            # Convert to RGB to ensure no alpha channel issues
                            clean_img = img.convert('RGB')
                            clean_img.save(file_path, 'JPEG')
                    except Exception:
                        raise ValueError("Corrupt JPEG data (missing EOF marker) and could not be fixed")

        result['retained_files'] = 1
    except Exception:
        # If PIL fails to read it, or it's unfixable, delete it
        file_path.unlink(missing_ok=True)
        result['deleted_unreadable'] = 1

    return result

def sanitize_dataset(dataset_root: Path, class_names: list[str], max_workers: int = CPU_THREADS) -> pd.DataFrame:
    """Multithreaded sanitization of the dataset directories."""
    require_dataset_root(dataset_root, class_names)
    rows = []

    for class_name in class_names:
        class_dir = dataset_root / class_name
        files = [file_path for file_path in class_dir.rglob('*') if file_path.is_file()]

        summary = {
            'total_files': 0,
            'deleted_zero_kb': 0,
            'deleted_666': 0,
            'deleted_unreadable': 0,
            'retained_files': 0,
        }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for file_result in executor.map(sanitize_file, files):
                for key, value in file_result.items():
                    summary[key] += value

        rows.append({'class_name': class_name, **summary})

    return pd.DataFrame(rows)

def get_dataloaders(dataset_root: Path):
    """Load raw image datasets from directory with an 80:20 split."""
    
    # Define ResNet50 transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root=str(dataset_root), transform=transform)
    
    # 80:20 split
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=CPU_THREADS, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=CPU_THREADS, 
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes
