"""
Training script for genre classification model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from model import create_model
from preprocessing import load_and_preprocess_audio, spec_augment
from config import TRAINING_CONFIG, MODEL_CONFIG, GENRE_LABELS, MODELS_DIR


class AudioDataset(Dataset):
    """Dataset class for audio files"""
    def __init__(self, file_list, labels, augment=False):
        """
        Args:
            file_list: List of audio file paths
            labels: List of corresponding label indices
            augment: Whether to apply data augmentation
        """
        self.file_list = file_list
        self.labels = labels
        self.augment = augment
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load and preprocess audio
        spec = load_and_preprocess_audio(self.file_list[idx])
        
        # Apply augmentation if training
        if self.augment:
            spec = spec_augment(spec)
        
        # Convert to tensor and add channel dimension
        spec = torch.FloatTensor(spec).unsqueeze(0)  # (1, n_mels, time_frames)
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return spec, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for specs, labels in tqdm(dataloader, desc="Training"):
        specs, labels = specs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item()
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for specs, labels in tqdm(dataloader, desc="Validation"):
            specs, labels = specs.to(device), labels.to(device)
            
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()
    
    return total_loss / len(dataloader), 100. * correct / total


def train_model(train_files, train_labels, val_files, val_labels, save_path=None):
    """
    Main training function
    
    Args:
        train_files: List of training audio file paths
        train_labels: List of training labels
        val_files: List of validation audio file paths
        val_labels: List of validation labels
        save_path: Path to save the best model
        
    Returns:
        dict: Training history
    """
    # Setup device
    device = torch.device(TRAINING_CONFIG["device"])
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset(train_files, train_labels, augment=True)
    val_dataset = AudioDataset(val_files, val_labels, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = create_model(
        n_mels=MODEL_CONFIG["input_shape"][0],
        n_classes=MODEL_CONFIG["n_classes"]
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(TRAINING_CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        # Save metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'genre_labels': GENRE_LABELS,
                }, save_path)
                print(f"Model saved to {save_path}")
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return history


if __name__ == "__main__":
    # Example usage - you need to provide your FMA dataset paths
    print("Training script ready. Please prepare your FMA dataset and update file paths.")
    print("\nExpected data format:")
    print("- train_files: List of paths to training audio files")
    print("- train_labels: List of genre indices (0-15)")
    print("- val_files: List of paths to validation audio files")
    print("- val_labels: List of genre indices (0-15)")
    print(f"\nGenre mapping: {dict(enumerate(GENRE_LABELS))}")
