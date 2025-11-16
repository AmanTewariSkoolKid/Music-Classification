"""
CNN model for music genre classification
"""
import torch
import torch.nn as nn


class GenreClassifier(nn.Module):
    """
    Convolutional Neural Network for genre classification from log-mel spectrograms
    """
    def __init__(self, n_mels=128, n_classes=16):
        super(GenreClassifier, self).__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames)
            
        Returns:
            torch.Tensor: Logits of shape (batch, n_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_model(n_mels=128, n_classes=16):
    """
    Factory function to create a genre classifier model
    
    Args:
        n_mels: Number of mel frequency bins
        n_classes: Number of genre classes
        
    Returns:
        GenreClassifier: Initialized model
    """
    return GenreClassifier(n_mels=n_mels, n_classes=n_classes)
