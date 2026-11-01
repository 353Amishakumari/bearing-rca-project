# # This code builds a 1D-CNN model to detect bearing faults from vibration signals.
import torch
import torch.nn as nn

class Bearing1DCNN(nn.Module): # Raw vibration,Basic patterns → 16 features,Deeper patterns → 32 features,Higher-level patterns → 64 features.
    def __init__(self, num_classes=4):
        super(Bearing1DCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(16),## Extracts deeper vibration patterns using 32 filters from 16 input features
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x