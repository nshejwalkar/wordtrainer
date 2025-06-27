import glob, cv2, pytesseract, os
from tqdm import tqdm

PICS_DIR        = '../../models_data/pics'
RAW_TILE_DIR    = '../../models_data/train/_UNSURE'
TRAIN_DIR       = '../../models_data/train'
UNSURE_DIR      = os.path.join(TRAIN_DIR, '_UNSURE')
REPORT_PATH     = '../../models_data/augmentation_report.csv'

TEMPLATE_SIZE   = 90     # every tile becomes 90×90 px
BOARD_SIZE      = 5      # Word-Hunt board is always 5×5
AUGMENT_TARGET  = 100    # desired tiles / class (after augment)

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
BLUR_P = THICKEN_P = THIN_P = 0.40
NOISE_P = 0.30
BC_P    = 0.40

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
TESS_CONFIG = '-l eng --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'


for tile_path in tqdm(glob.glob(f'{RAW_TILE_DIR}/*.png')[:10], desc='autosort'):
   img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
   txt = pytesseract.image_to_data(img, config=TESS_CONFIG, output_type=pytesseract.Output.DICT)
   # print(f'TESSERACT TEXT IS {txt['text']}')
   # print(f'TESSERACT CONF IS {txt['conf']}')
   if len(txt['text'])==0 or txt['text'][-1]=='':
      print("text is null")
      target = UNSURE_DIR
   else:
      char  = txt['text'][-1].upper()
      conf  = int(txt['conf'][-1])
      if char.isalpha() and conf >= CONF_THRESH:
         target = os.path.join(TRAIN_DIR, char)
      else:
         target = UNSURE_DIR

   print('-----------------------------------------')