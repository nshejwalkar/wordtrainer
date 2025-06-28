import torch.nn as nn

WEIGHTS   = 'board_cnn_100.pt'

class TinyCNN(nn.Module):
   def __init__(self, n_classes):
      super().__init__()
      self.features = nn.Sequential(
         nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
         nn.MaxPool2d(2),                # 90 → 45
         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
         nn.MaxPool2d(2),                # 45 → 22
         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
      )
      self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pool  → (64,1,1)
      self.classifier = nn.Linear(64, n_classes)

   def forward(self, x):
      x = self.features(x)
      x = self.pool(x).flatten(1)
      return self.classifier(x)

