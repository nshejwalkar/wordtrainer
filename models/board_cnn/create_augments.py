#!/usr/bin/env python3
# create_data.py  –  extract ▸ autosort ▸ augment
# 3-space indentation everywhere
# ============================================================

import os, cv2, csv, glob, random, shutil, argparse
import numpy as np
from tqdm import tqdm
import pytesseract

# ------------------------------------------------------------
# CONSTANTS  (edit here)
# ------------------------------------------------------------
PICS_DIR        = '../../models_data/pics'
RAW_TILE_DIR    = '../../models_data/tiles_raw'
TRAIN_DIR       = '../../models_data/train'
UNSURE_DIR      = os.path.join(TRAIN_DIR, '_UNSURE')
REPORT_PATH     = '../../models_data/augmentation_report.csv'

TEMPLATE_SIZE   = 90     # every tile becomes 90×90 px
BOARD_SIZE      = 5      # Word-Hunt board is always 5×5
AUGMENT_TARGET  = 750    # desired tiles / class (after augment)

CONF_THRESH     = 60     # Tesseract confidence cut-off (< goes to _UNSURE)

MAGIC = {   # relative board ROI for iPhone-13 screenshots
   'board': {
      'ytop':    0.42552083333333335,
      'ybottom': 0.7927083333333333,
      'xleft':   0.10247747747747747,
      'xright':  0.8975225225225225,
   }
}

# probability of each augmentation
ZOOM_P = 0.60
BC_P = THICKEN_P = NOISE_P = 0.50
THIN_P = BLUR_P = 0.20

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
TESS_CONFIG = '-l eng --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def magic_to_pixels(shape):
   h, w = shape[:2]
   m = MAGIC['board']
   return (int(h*m['ytop']), int(h*m['ybottom']),
           int(w*m['xleft']), int(w*m['xright']))

def autocrop(img, thr=10):
   g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   _, bw = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
   nz = cv2.findNonZero(bw)
   return img if nz is None else img[cv2.boundingRect(nz)[1]:cv2.boundingRect(nz)[1]+cv2.boundingRect(nz)[3],
                                     cv2.boundingRect(nz)[0]:cv2.boundingRect(nz)[0]+cv2.boundingRect(nz)[2]]

def pad_resize(tile, size=TEMPLATE_SIZE):
   tile = autocrop(tile)
   h, w = tile.shape[:2]
   if max(h, w) > size:
      s = size / max(h, w)
      tile = cv2.resize(tile, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
      h, w = tile.shape[:2]
   pad_t = (size-h)//2; pad_b = size-h-pad_t
   pad_l = (size-w)//2; pad_r = size-w-pad_l
   return cv2.copyMakeBorder(tile, pad_t, pad_b, pad_l, pad_r,
                             cv2.BORDER_CONSTANT, value=0)

# ------------------------------------------------------------
# phase: extract
# ------------------------------------------------------------
def extract_tiles():
   os.makedirs(RAW_TILE_DIR, exist_ok=True)
   total = 0
   for img_path in tqdm(glob.glob(f'{PICS_DIR}/*'), desc='extract'):
      img = cv2.imread(img_path)
      if img is None: continue
      y1,y2,x1,x2 = magic_to_pixels(img.shape)
      board = cv2.cvtColor(img[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
      _, bw = cv2.threshold(board,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      th, tw = bw.shape[0]//BOARD_SIZE, bw.shape[1]//BOARD_SIZE
      for r in range(BOARD_SIZE):
         for c in range(BOARD_SIZE):
            tile = pad_resize(bw[r*th:(r+1)*th, c*tw:(c+1)*tw])
            fname = f"{os.path.splitext(os.path.basename(img_path))[0]}_{r}{c}.png"
            cv2.imwrite(os.path.join(RAW_TILE_DIR,fname), tile)
            total += 1
   print(f"✅ extracted {total} tiles → {RAW_TILE_DIR}")

# ------------------------------------------------------------
# phase: autosort  (Tesseract one-char guess)
# ------------------------------------------------------------
def autosort_tiles():
   # create A..Z folders & _UNSURE
   for L in [chr(ord('A')+i) for i in range(26)]:
      os.makedirs(os.path.join(TRAIN_DIR,L), exist_ok=True)
   os.makedirs(UNSURE_DIR, exist_ok=True)

   moved, unsure = 0, 0
   for tile_path in tqdm(glob.glob(f'{RAW_TILE_DIR}/*.png'), desc='autosort'):
      img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
      txt = pytesseract.image_to_data(img, config=TESS_CONFIG, output_type=pytesseract.Output.DICT)
      if len(txt['text'])==0 or txt['text'][-1]=='':
         target = UNSURE_DIR
         unsure += 1
      else:
         char  = txt['text'][-1].upper()
         conf  = int(txt['conf'][-1])
         if char.isalpha() and conf >= CONF_THRESH:
            target = os.path.join(TRAIN_DIR, char)
         else:
            target = UNSURE_DIR
            unsure += 1

      # ensure unique name
      base = os.path.basename(tile_path)
      out  = os.path.join(target, base)
      cnt  = 1
      while os.path.exists(out):
         out = os.path.join(target, f"{os.path.splitext(base)[0]}_{cnt}.png")
         cnt += 1
      shutil.move(tile_path, out)
      moved += 1
   print(f"✅ autosorted {moved} tiles  |  moved to _UNSURE: {unsure}")

# ------------------------------------------------------------
# phase: augment (unchanged except it now assumes folders exist)
# ------------------------------------------------------------

def zoom_at(img, zoom=1): 
   # print(img.shape)
   # print(img.shape[:-1])
   # print([ i/2 for i in img.shape[:-1] ])

   h, w = img.shape[:2]
   # Compute crop size
   new_h, new_w = int(h / zoom), int(w / zoom)
   y1 = (h - new_h) // 2
   x1 = (w - new_w) // 2
   cropped = img[y1:y1+new_h, x1:x1+new_w]
   return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def random_aug(tile):
   if random.random()<THICKEN_P:
      tile=cv2.dilate(tile,np.ones((2,2),np.uint8),1)
   if random.random()<THIN_P:
      tile=cv2.erode(tile,np.ones((3,3),np.uint8),1)
   if random.random()<BLUR_P:
      tile=cv2.GaussianBlur(tile,(3,3),0)
   if random.random()<BC_P:  # brightness/contrast
      tile=cv2.convertScaleAbs(tile,alpha=random.uniform(0.9,1.1),beta=random.randint(-10,10))
   if random.random()<NOISE_P:
      n=np.random.normal(0,3,tile.shape).astype(np.int16)
      tile=np.clip(tile.astype(np.int16)+n,0,255).astype(np.uint8)
   z=1
   if random.random()<ZOOM_P:
      z = random.random()*0.5 + 1.1
      tile = zoom_at(tile, zoom=z)  # between 1 and 1.5
   return (tile,z)

def augment():
   letters=[chr(ord('A')+i) for i in range(26)]
   rows=[['letter','orig','created','total']]
   for L in letters:
      cdir=os.path.join(TRAIN_DIR,L)
      os.makedirs(cdir,exist_ok=True)
      imgs = glob.glob(f'{cdir}/*.png')                             
      originals = [f for f in imgs if 'augmented' not in f]
      orig=len(imgs); created=0
      with tqdm(total=max(0,AUGMENT_TARGET-orig),desc=f'augment {L}') as pb:
         while len(imgs)<AUGMENT_TARGET and imgs:
            src=cv2.imread(random.choice(originals),cv2.IMREAD_GRAYSCALE)
            aug,z=random_aug(src)
            name=f'{L}_{len(imgs):04d}_zoom_{z:.3f}_augmented.png'
            cv2.imwrite(os.path.join(cdir,name),aug)
            imgs.append(os.path.join(cdir, name)); created+=1; pb.update(1)
      rows.append([L,orig,created,len(imgs)])
   with open(REPORT_PATH,'w',newline='') as f: csv.writer(f).writerows(rows)
   print(f'✅ augment done → {REPORT_PATH}')

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == '__main__':
   ap=argparse.ArgumentParser()
   ap.add_argument('--phase',default='augment',choices=['extract','autosort','augment'])
   args=ap.parse_args()
   if args.phase=='extract':  extract_tiles()
   elif args.phase=='autosort':autosort_tiles()
   elif args.phase=='augment': augment()
