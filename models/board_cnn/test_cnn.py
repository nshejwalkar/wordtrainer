#!/usr/bin/env python3
# test_cnn.py – quick accuracy sanity-check
# 3-space indentation everywhere
# -------------------------------------------------------------

import os, torch, argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, ToTensor, Normalize
from collections import defaultdict

# ---------- paths -----------------------------------------------------------
WEIGHTS   = 'board_cnn.pt'                 # adjust if stored elsewhere
DATA_DIR  = '../../models_data/train'      # root that has A … Z folders

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SZ  = 128

# ---------- TinyCNN definition (same as training script) --------------------
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

# ---------- dataset ---------------------------------------------------------
tfm = Compose([Grayscale(1),
               ToTensor(),
               Normalize([0.5], [0.5])])

class NoUnsure(ImageFolder):
   def find_classes(self, directory):
      cls = [d.name for d in os.scandir(directory)
             if d.is_dir() and d.name != '_UNSURE']
      cls.sort(); return cls, {c:i for i,c in enumerate(cls)}

ds  = NoUnsure(DATA_DIR, transform=tfm)
ld  = DataLoader(ds, batch_size=BATCH_SZ, shuffle=False)

# ---------- load model ------------------------------------------------------
net = TinyCNN(len(ds.classes)).to(DEVICE)
net.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
net.eval()

# ---------- evaluation ------------------------------------------------------
total, correct = 0, 0
per_cls_hit  = defaultdict(int)
per_cls_total = defaultdict(int)

with torch.no_grad():
   for xb, yb in ld:
      xb, yb = xb.to(DEVICE), yb.to(DEVICE)
      pred   = net(xb).argmax(1)
      correct += (pred == yb).sum().item()
      total   += yb.size(0)

      for t, p in zip(yb.cpu(), pred.cpu()):
         per_cls_total[t.item()] += 1
         if t == p:
            per_cls_hit[t.item()] += 1

overall_acc = correct / total * 100
print(f"\nOverall accuracy on {total} tiles : {overall_acc:.2f} %")

print("\nPer-class accuracy:")
for idx, cls in enumerate(ds.classes):
   acc = 100 * per_cls_hit[idx] / per_cls_total[idx]
   print(f"  {cls:>2} : {acc:6.2f} %  ({per_cls_hit[idx]}/{per_cls_total[idx]})")
