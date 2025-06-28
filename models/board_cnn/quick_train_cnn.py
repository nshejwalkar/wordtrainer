#!/usr/bin/env python3
# train_cnn_debug.py – minimal, LOUD, over-fit test
# 3-space indentation everywhere
# ============================================================

import os, random, cv2, torch, argparse
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Grayscale, Compose
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
from pathlib import Path

DATA_DIR  = '../../models_data/train'        # A … Z folders (no _UNSURE)
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
LR        = 1e-1
BATCH_SZ  = 32
STEPS     = 100                               # lower stops when acc = 1.0

# ------------------------------------------------------------
# dataset (first 64 tiles only)
# ------------------------------------------------------------
class NoUnsure(ImageFolder):
   def find_classes(self, directory):
      cls = [d.name for d in os.scandir(directory)
             if d.is_dir() and d.name != '_UNSURE']
      cls.sort(); return cls, {c:i for i,c in enumerate(cls)}

tfm = Compose([Grayscale(1), ToTensor()])     # 0-1 float, 1×H×W
full = NoUnsure(DATA_DIR, transform=tfm)

# --- grab 2 tiles / class so we see every letter -------------------
cls_to_indices = {c: [] for c in full.classes}
for idx, (_, lab) in enumerate(full.samples):
   if len(cls_to_indices[full.classes[lab]]) < 2:
      cls_to_indices[full.classes[lab]].append(idx)
   if all(len(v) == 2 for v in cls_to_indices.values()):
      break

flat_idx = [i for sub in cls_to_indices.values() for i in sub]
random.shuffle(flat_idx)                        # make order random
subset = Subset(full, flat_idx)                 # 52 tiles


# visual check ------------------------------------------------
# plt.figure(figsize=(6,3))
# for i in range(8):
#    img, lab = subset[i]
#    plt.subplot(2,4,i+1); plt.imshow(img[0], cmap='gray')
#    plt.title(full.classes[lab]); plt.axis('off')
# plt.tight_layout(); plt.show()

# for i,(img,lab) in enumerate(subset):
#    plt.subplot(4,13,i+1)
#    plt.imshow(img[0],cmap='gray')
#    plt.title(f"{i}:{full.classes[lab]}",fontsize=6)
#    plt.axis('off')
# plt.tight_layout(); plt.show()

# split 32 / 32 ----------------------------------------------
train_set = Subset(subset, range(32))
val_set   = Subset(subset, range(32,64))
train_ld  = DataLoader(train_set, batch_size=BATCH_SZ, shuffle=True)
val_ld    = DataLoader(val_set,   batch_size=BATCH_SZ, shuffle=False)

print(f"Loaded {len(train_set)} train + {len(val_set)} val samples")
print("Sample dtype / range :", train_set[0][0].dtype,
      train_set[0][0].min().item(), train_set[0][0].max().item())
print("Label example        :", train_set[0][1], type(train_set[0][1]))

# ------------------------------------------------------------
# tiny 3-conv net
# ------------------------------------------------------------
class TinyCNN(nn.Module):
   def __init__(self, n):
      super().__init__()
      self.conv1 = nn.Conv2d(1,16,3,padding=1)
      self.conv2 = nn.Conv2d(16,32,3,padding=1)
      self.conv3 = nn.Conv2d(32,64,3,padding=1)
      self.pool  = nn.AdaptiveAvgPool2d(1)
      self.fc    = nn.Linear(64, n)
   def forward(self,x):
      x = torch.relu(self.conv1(x)); x = torch.max_pool2d(x,2)
      x = torch.relu(self.conv2(x)); x = torch.max_pool2d(x,2)
      x = torch.relu(self.conv3(x)); x = self.pool(x).flatten(1)
      return self.fc(x)

class LogitPix(nn.Module):
   def __init__(self,n_cls): 
      super().__init__()
      self.w = nn.Linear(90*90, n_cls)
   def forward(self,x): return self.w(x.view(x.size(0), -1))

net = TinyCNN(len(full.classes)).to(DEVICE)
# net = LogitPix(len(full.classes)).to(DEVICE)
opt = optim.SGD(net.parameters(), lr=1e-1)          # big but stable for SGD
print(f"Params: {sum(p.numel() for p in net.parameters()):,}")

# ------------------------------------------------------------
# sanity probe 1 – forward pass shape
# ------------------------------------------------------------
x0, y0 = train_set[0]
logits0 = net(x0.unsqueeze(0).to(DEVICE))
print("Single forward logits shape:", logits0.shape)           # [1,26]

# sanity probe 2 – CE loss should be ~3.26 for random logits
criterion = nn.CrossEntropyLoss()
print("Random-init CE loss       :", criterion(logits0, torch.tensor([y0]).to(DEVICE)).item())

# sanity probe 3 – gradient non-zero ?
criterion(logits0, torch.tensor([y0]).to(DEVICE)).backward()
print("Grad mean fc.weight       :", net.fc.weight.grad.abs().mean().item())

# reset weights so we train from scratch ---------------------
def reinit(m):
   if isinstance(m, (nn.Conv2d, nn.Linear)):
      nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)
# net.apply(reinit)
net.apply(reinit)
for p in net.parameters():                      # zero Adam moments after re-init
   if hasattr(p, 'grad'): p.grad = None
opt = optim.Adam(net.parameters(), lr=LR)

# ------------------------------------------------------------
# training loop (no AMP, no scheduler, just print everything)
# ------------------------------------------------------------
def run(loader, train=True):
   net.train(train)
   total, correct, loss_sum = 0, 0, 0.0
   for xb, yb in loader:
      xb, yb = xb.to(DEVICE), yb.to(DEVICE)
      logits = net(xb)
      loss   = criterion(logits, yb)
      if train:
         opt.zero_grad(); loss.backward(); opt.step()

      loss_sum += loss.item()*xb.size(0)
      correct  += (logits.argmax(1)==yb).sum().item()
      total    += xb.size(0)

      # log statistics of the batch
      print(f"[batch] loss {loss.item():.3f}  "
            f"logit μ {logits.mean().item():+.3f} σ {logits.std().item():.3f}")

   print(f"{'train' if train else 'val  '} "
         f"loss {loss_sum/total:.3f}  acc {(correct/total):.2%}")

for step in range(STEPS):
   print(f"\n=== STEP {step} =========================")
   run(train_ld, train=True)

   # stop early when we over-fit
   with torch.no_grad():
      n_correct = 0
      for xb,yb in train_ld:
         xb,yb = xb.to(DEVICE), yb.to(DEVICE)
         n_correct += (net(xb).argmax(1)==yb).sum().item()
      if n_correct == len(train_set):
         print(">>> reached 100 % on tiny train set – done")
         break

print("\nFinished debugging run.")
print("If you never reached 100 % check the printed logits, gradients, labels.")
