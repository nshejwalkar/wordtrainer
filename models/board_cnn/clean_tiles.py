import cv2, os
from tqdm import tqdm
import numpy as np

ROOT = '../../models_data/train'      # A/…/Z folders
SIZE = 90                             # final canvas size

for cls in os.listdir(ROOT):
   cls_dir = os.path.join(ROOT, cls)
   if not os.path.isdir(cls_dir): continue
   for fname in tqdm(os.listdir(cls_dir), desc=cls):
      if not fname.lower().endswith('.png'): 
         print(f'didnt exist: {cls_dir}/{fname}')
         continue
      p = os.path.join(cls_dir, fname)
      img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

      # --- Step-1: threshold & keep largest component -------------
      _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
      n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
      if n <= 1:   # no foreground
         continue
      biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
      mask = (labels == biggest).astype(np.uint8) * 255

      # --- Step-2: crop to glyph, paste on new canvas -------------
      y,x,h,w = cv2.boundingRect(mask)     # note: (x,y,w,h)
      glyph   = mask[y:y+h, x:x+w]

      canvas  = np.zeros((SIZE, SIZE), dtype=np.uint8)
      pad_y   = (SIZE - h) // 2
      pad_x   = (SIZE - w) // 2
      canvas[pad_y:pad_y+h, pad_x:pad_x+w] = glyph

      cv2.imwrite(p, canvas)
