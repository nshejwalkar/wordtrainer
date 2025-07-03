# train_cnn.py - train a tiny A-Z classifier for Word-Hunt tiles
# =============================================================
#  • Expects the 26 folders produced in phase-augment:
#      ../../models_data/train/A  …  ../../models_data/train/Z
#  • Saves best weights to    models/board_cnn/board_cnn.pt
#  • Uses a compact 3-Conv CNN (~110k params)
#  • Mixed-precision & GPU friendly
#  • 3-space indentation everywhere (PEP-8-ish but narrower)
# =============================================================

import glob
import os, random, cv2
from pathlib import Path

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -------------------------------------------------------------
# CONSTANTS – tweak here
# -------------------------------------------------------------
DATA_DIR        = '../../models_data/train'          # root with A..Z/
WEIGHTS_OUT     = 'board_cnn.pt'                     # saved in same folder as script
BATCH_SIZE      = 32
EPOCHS          = 20
LR              = 1e-2
SCHED_FACTOR    = 0.3
PATIENCE        = 2
EARLY_STOP_PATIENCE = 3                              # epochs w/o val-loss improvement
IMG_SIZE        = 90                                 # tiles are 90×90
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 21  # Pick any integer
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------
# Dataset & transforms
# -------------------------------------------------------------
transform = Compose([
   transforms.Grayscale(num_output_channels=1),  # convert RGB to grayscale (1 channel)
   ToTensor(),                          # HWC [0-255] → CHW [0-1]
   Normalize(mean=[0.5], std=[0.5])     # map white≈1 → ~1, black≈0 → ~-1
])

class ImageFolderNoUnsure(ImageFolder):
   def find_classes(self, directory):
      classes = [d.name for d in os.scandir(directory) if d.is_dir() and d.name != '_UNSURE']
      classes.sort()
      class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
      return classes, class_to_idx

dataset = ImageFolderNoUnsure(DATA_DIR, transform=transform)
print("Classes:", dataset.classes)
print("Transform: ", dataset.transform)
num_classes = len(dataset.classes)                 # should be 26
print(f"Loaded {len(dataset)} samples, {num_classes} classes from {DATA_DIR}")


# train/val split 90/10
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

tiny_loader = DataLoader(torch.utils.data.Subset(train_ds, list(range(64))), batch_size=32)
# subset = torch.utils.data.Subset(dataset, list(range(64)))
# loader = DataLoader(subset, batch_size=64)


img, label = dataset[0]
print("Shape:", img.shape)
print("Pixel min/max:", img.min().item(), img.max().item())

# Show a grid of 8 random training images
# batch = next(iter(train_loader))
# images, labels = batch
# grid = torchvision.utils.make_grid(images[:8], nrow=4, normalize=True)
# plt.imshow(grid.permute(1, 2, 0))
# plt.title(f"Labels: {[dataset.classes[l] for l in labels[:8]]}")
# plt.show()

# for i in range(8):
#    img, label = tiny_loader.dataset[i]
#    plt.subplot(2, 4, i+1)
#    plt.imshow(img[0], cmap='gray')  # single channel
#    plt.title(dataset.classes[label])
#    plt.axis('off')
# plt.tight_layout()
# plt.show()


# -------------------------------------------------------------
# Model – tiny CNN (~110k params)
# -------------------------------------------------------------
class TinyCNN(nn.Module):
   def __init__(self, n_classes):
      super().__init__()
      self.features = nn.Sequential(
         nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
         nn.MaxPool2d(2),                # 90 → 45
         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
         nn.MaxPool2d(2),                # 45 → 22
         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
         nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
         # still → (128, 22, 22) before pooling
      )
      self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pool  → (64,1,1)
      self.classifier = nn.Linear(128, n_classes)

   def forward(self, x):
      x = self.features(x)
      x = self.pool(x).flatten(1)
      return self.classifier(x)

model = TinyCNN(num_classes).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
              optimizer, factor=SCHED_FACTOR, patience=PATIENCE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # very light smoothing (avoid one-hot -> turns them soft)
scaler = torch.amp.GradScaler('cuda')  # mixed precision

# -------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------

def plot_confusion_matrix(model, loader, class_names):
   model.eval()
   all_preds, all_labels = [], []
   with torch.no_grad():
      for inputs, labels in loader:
         inputs = inputs.to(DEVICE)
         outputs = model(inputs)
         preds = outputs.argmax(1).cpu().numpy()
         all_preds.extend(preds)
         all_labels.extend(labels.numpy())

   cm = confusion_matrix(all_labels, all_preds)
   disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
   fig, ax = plt.subplots(figsize=(10, 10))
   disp.plot(ax=ax, cmap='Blues', colorbar=False)
   plt.title("Validation Confusion Matrix")
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.show()

def run_epoch(loader, train=True):
   model.train(train)
   running_loss, correct, total = 0.0, 0, 0
   loop = tqdm(loader, leave=False)
   
   for inputs, labels in loop:
      inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
      # with torch.amp.autocast('cuda'):
      if train:
         optimizer.zero_grad(set_to_none=True)
      outputs = model(inputs)
      # print(outputs.shape)
      loss = criterion(outputs, labels)
      if train:
         scaler.scale(loss).backward()
         scaler.step(optimizer)
         scaler.update()
      preds = outputs.argmax(1)
      running_loss += loss.item() * inputs.size(0)
      correct += (preds == labels).sum().item()
      total += labels.size(0)
      loop.set_description('train' if train else 'val ')
      loop.set_postfix(loss=loss.item())

   return running_loss / total, correct / total

best_val_loss = float('inf')
patience = EARLY_STOP_PATIENCE

# for epoch in range(50):
#    loss, acc = run_epoch(tiny_loader, train=True)
#    print(f"[tiny] epoch {epoch}  loss={loss:.4f}  acc={acc:.3f}")

def full_train():
   global SEED  # made this global to avoid scope error
   print(f'DEVICE is {DEVICE}')
   best_val_loss = float('inf')
   # did we already start training before?
   start = 1
   pt_files = glob.glob('./checkpoints/checkpoint_epoch_*.pt')
   if pt_files:
      latest = max(pt_files, key=lambda f: int(f.split('.')[-2].split('_')[-1]))
      checkpoint = torch.load(latest)
      print(checkpoint)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      start = checkpoint['epoch'] + 1
      print(f"starting at {start}")
      # SEED = checkpoint['seed']

   for epoch in range(start, EPOCHS+1):
      print(f"\nEpoch {epoch}/{EPOCHS}")
      train_loss, train_acc = run_epoch(train_loader, train=True)
      with torch.no_grad():
         val_loss, val_acc = run_epoch(val_loader, train=False)

      print(f"  train-loss {train_loss:.4f}  acc {train_acc:.3f}  |  "
            f"val-loss {val_loss:.4f}  acc {val_acc:.3f}")

      # save model weights every epoch
      torch.save({
         'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'seed': SEED,
      }, f'./checkpoints/checkpoint_epoch_{epoch}.pt')

      # Early stopping & checkpoint
      if val_loss < best_val_loss - 1e-4:
         best_val_loss = val_loss
         patience = EARLY_STOP_PATIENCE
         torch.save(model.state_dict(), WEIGHTS_OUT)
         print(f"  ✅ Saved best model to {WEIGHTS_OUT}")
      else:
         patience -= 1
         if patience == 0:
            print("  🛑 Early stopping (no val improvement)")
            break

full_train()

print("\nTraining complete.")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Weights saved to: {Path(WEIGHTS_OUT).resolve()}")
print("About to plot confusion matrix:")

# checkpoint = torch.load('./checkpoints/checkpoint_epoch_17.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
plot_confusion_matrix(model, val_loader, dataset.classes)

# -------------------------------------------------------------
# Inference helper (optional import-able)
# -------------------------------------------------------------

def load_model(weights_path=WEIGHTS_OUT):
   m = TinyCNN(num_classes)
   m.load_state_dict(torch.load(weights_path, map_location='cpu'))
   m.eval()
   return m

def predict_tile(np_img, model, device='cpu'):
   """np_img: H×W uint8 (white glyph on black). Returns int label."""
   t = torch.from_numpy(np_img).unsqueeze(0).float() / 255.0  # H W → 1 H W
   t = Normalize(0.5, 0.5)(t)  # same norm as training
   with torch.no_grad():
      logits = model(t.to(device))
      return logits.argmax(1).item()  # 0..25