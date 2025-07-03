import time
print("Booting...")
t0 = time.time()

import cv2, pytesseract
import numpy as np
import subprocess
import os, glob, torch
import torch.nn as nn
import torchvision.transforms as transforms

print("Done with imports:", time.time() - t0)


# VERY hardcoded for now. works for my phone (iPhone 13), might work for others
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
sample_video = './uploads/ScreenRecording_06-15-2025 16-33-15_1.mp4'
# sample_video = './uploads/ScreenRecording_06-23-2025 19-51-01_1.mov'
sample_video_120 = './uploads/iphone12_120.mp4'
sample_image = os.path.abspath('../models_data/pics/IMG_6358.PNG')

custom_config = '-l eng --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"'
custom_config_words = '-l eng --psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+"'

WEIGHTS   = './model_weights/board_cnn_4L_2.pt'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

# need relatives. pixels never guaranteed to match
MAGIC = {
   'time': {'ytop': 0.17604166666666668, 'ybottom': 0.203125, 'xleft': 0.6981981981981982, 'xright': 0.9234234234234234},
   'board': {'ytop': 0.42552083333333335, 'ybottom': 0.7927083333333333, 'xleft': 0.10247747747747747, 'xright': 0.8975225225225225},
   'word_count': {'ytop': 0.096875, 'ybottom': 0.11510416666666666, 'xleft': 0.5, 'xright': 0.6295045045045045},
   'words': {'ytop': 0.3578, 'ybottom': 0.39479, 'xleft': 0.17229, 'xright': 0.81869},
   'score': {'ytop': 0.11614583333333334, 'ybottom': 0.14895833333333333, 'xleft': 0.6216216216216216, 'xright': 0.8975225225225225}
}

def transcode_to_120fps(video_path, output_path):
   print("Transcoding video to 120fps CFR...")

   command = [
      'ffmpeg',
      '-i', video_path,
      '-filter:v', 'fps=120',  # Force 120 fps
      '-vsync', 'cfr',         # Enforce constant frame rate
      '-c:v', 'libx264',       # Use x264 for compatibility
      '-preset', 'veryfast',   # Speed-vs-size tradeoff
      '-crf', '18',            # Visually lossless quality
      '-movflags', '+faststart',  # Optimize for playback
      output_path
   ]

   subprocess.run(command, check=True)
   print(f"✅ Transcoded video saved to {output_path}")


# converts relative coordinates to pixel coordinates based on video resolution
# returns a dictionary with the same structure as magic, but with absolute pixel values
# ONLY for videos. INDETERMINATE BEHAVIOR FOR IMAGES
def _magic_to_pixels(video_path, debug=True):
   cap = cv2.VideoCapture(video_path)
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   if debug:
      print(f"Video dimensions: {width}x{height}")
   cap.release()
   pixels = {}
   for key in MAGIC.keys():
      pixels[key] = {k:(int(v*width) if k.startswith('x') else int(v*height)) for k,v in MAGIC[key].items()}  # ???
      # for k,v in magic[key].items():
      #    if k.startswith('x'):
      #       pixels[key][k] = int(v*width)
      #    else:
      #       pixels[key][k] = int(v*height)
   return pixels

# this is for images
def _magic_to_pixels_image(image_path, debug=True):
   img = cv2.imread(image_path)
   height, width = img.shape[:2]
   if debug:
      print(f"Image dimensions: {height}x{width}")
   
   pixels = {}
   for key in MAGIC.keys():
      pixels[key] = {k:(int(v*width) if k.startswith('x') else int(v*height)) for k,v in MAGIC[key].items()}  # ???

   return pixels

# extracts the region of interest (ROI) from the frame based on the magic coordinates
# interest is a key in the magic dictionary: 'time', 'board', 'word_count', 'score'
# if the roi.shape is returning (0,0,3) or 0x0: THE PROBLEM IS IN _MAGIC_TO_PIXELS() NOT HERE
def _get_roi(frame: np.ndarray, magic_pix, interest, debug=True):
   if debug:
      print(frame.shape)
   interest_subdict = magic_pix[interest]
   ytop = interest_subdict['ytop']
   ybottom = interest_subdict['ybottom']
   xleft = interest_subdict['xleft']
   xright = interest_subdict['xright']
   roi = frame[ytop:ybottom, xleft:xright]
   # roi = frame[interest_subdict['ytop']:interest_subdict['ybottom'], interest_subdict['xleft']:interest_subdict['xright']]
   if debug:
      print(roi.shape)
   return roi

# gets the nth frame from the video
def _get_frame(video_path, number):
   cap = cv2.VideoCapture(video_path)
   cap.set(cv2.CAP_PROP_POS_FRAMES, number)
   _, frame = cap.read()
   return frame

# applies preprocessing to the ROI to prepare it for OCR
def _apply_preprocessing_board(roi, modality='video', shape=(1920, 888, 3), debug=True):  # somehow this works 
   if debug:
      cv2.imshow('roi', roi)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
   # resize image to 1920x888
   height, width = shape[0], shape[1]
   if debug:
      print(f'height is {height}')
      print(f'width is {width}')
      print(f'fx is {888/width}')
      print(f'fy is {1920/height}')
   resized = cv2.resize(roi, None, fx=888/width, fy=1920/height, interpolation=cv2.INTER_AREA)
   if debug:
      print(f'new shape is {resized.shape}')
   if debug:
      cv2.imshow('roi', resized)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
   # make gray
   gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
   if debug:
      cv2.imshow('gray', gray)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
   equalized = cv2.equalizeHist(gray)
   if debug:
      cv2.imshow('equalized', equalized)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

   # binary threshold: anything less than 21 becomes white. 21-255 becomes black
   if modality == 'video':
      _, binary = cv2.threshold(equalized, 21, 255, cv2.THRESH_BINARY_INV)
   else:
      _, binary = cv2.threshold(equalized, 5, 255, cv2.THRESH_BINARY_INV)
   th2 = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 0)

   # close letters to get rid of artifacts
   kernel = np.ones((5,5),np.uint8)
   closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
   if debug:
      cv2.imshow('closing', closing)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
 
   # contours
   closing_contours = closing.copy()
   items = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   contours = items[0] if len(items) == 2 else items[1]
   cv2.drawContours(closing_contours, contours, -1, (0,0,255), 2)

   if debug:
      cv2.imshow('closing_contours', closing_contours)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

   # want to isolate black letters
   # lower = np.array([0,0,0])
   # upper = np.array([180,100,100])
   # mask = cv2.inRange(hsv, lower, upper)
   # print(closing_contours[:500])
   return closing_contours


def seek_thru_video(video_path, start_frame=0, max_display_width=1280):
   cap = cv2.VideoCapture(video_path)
   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

   current_frame = start_frame
   window_name = 'Frame'
   cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Make window resizable

   while True:
      # Make sure current_frame stays in bounds
      current_frame = max(0, min(current_frame, total_frames - 1))
      print(f"Current frame: {current_frame}/{total_frames - 1}")

      cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
      ret, frame = cap.read()

      if not ret:
         print(f"Frame {current_frame} could not be read.")
         break

      # Resize frame if it's too large
      height, width = frame.shape[:2]
      if width > max_display_width:
         scale = max_display_width / width
         frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

      cv2.imshow('Frame', frame)
      key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press

      if key == ord('q'):
         break
      elif key == ord(','):  # Left arrow or comma
         current_frame -= 1
      elif key == ord('.'):  # Right arrow or period
         current_frame += 1
      elif key == ord('k'):  
         current_frame -= 10
      elif key == ord('l'):  
         current_frame += 10
      elif key == ord('i'):  
         current_frame -= 100
      elif key == ord('o'):  
         current_frame += 100
      elif key == ord('8'):  
         current_frame -= 1000
      elif key == ord('9'):  
         current_frame += 1000

   cap.release()
   cv2.destroyAllWindows()


# binary searches between the first frame and the middle frame of the video
def get_frame_with_start(video_path):
   cap = cv2.VideoCapture(video_path)
   fnum = 0
   print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
   fps = cap.get(cv2.CAP_PROP_FPS)
   print(f'fps is {fps}')
   magic_pix = _magic_to_pixels(video_path)
   print(magic_pix)
   i = 0
   j = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2)  # start at the middle of the video
   
   while i < j:
      mid = (i + j) // 2
      cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
      ret, frame = cap.read()
      if not ret: break

      roi = _get_roi(frame, magic_pix, 'time')
      # cv2.imshow('roi', roi)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      # roi = _apply_preprocessing_board(roi)
      text = pytesseract.image_to_string(roi)

      fnum+=1
      print(fnum, end=' ')
      print(text)

      if "01:19" in text:  # found frame. close capture and return it.
         print(f"found at frame {fnum}, timestamp approx {fnum/fps} seconds")
         cap.release()
         return frame

   # didnt find shit
   cap.release()
   return None

   
def _split_into_tiles(board_roi, board_size):
   TARGET_SIZE = 90  # size of each tile after cropping and padding
   h, w = board_roi.shape[:2]
   tile_h, tile_w = h // board_size, w // board_size

   tiles = []
   for row in range(board_size):
      for col in range(board_size):
         x_start = col * tile_w
         y_start = row * tile_h
         tile = board_roi[y_start:y_start+tile_h, x_start:x_start+tile_w]
         # cv2.imshow('tile pre-crop', tile)
         # cv2.waitKey(0)
         # cv2.destroyAllWindows()

         # same exact thing as in template cleaning: bounding box + cropping (DOESNT WORK WITH ARTIFACTS)
         # _, thresh = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY)
         # coords = cv2.findNonZero(thresh)
         _, bw  = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY)
         cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         coords = max(cnts, key=cv2.contourArea)  # choose contour with the largest area (NO MORE ARTIFACTS!)
         x, y, w, h = cv2.boundingRect(coords)

         # check what this shit above actually does?
         if len(tile.shape) == 2:  # grayscale → convert to BGR
            tile_vis = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
         else:
            tile_vis = tile.copy()
         cv2.rectangle(tile_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
         # cv2.imshow("Bounding Box", tile_vis)
         # cv2.waitKey(0)
         # cv2.destroyAllWindows()

         # resize to 90x90 (need to crop, not pad)
         cropped = tile[y:y+h, x:x+w]

         # cv2.imshow('tile post-crop', cropped)
         # cv2.waitKey(0)
         # cv2.destroyAllWindows()
         
         if cropped.shape[0] > TARGET_SIZE or cropped.shape[1] > TARGET_SIZE:
            padded = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))
         else:
            pad_x = (TARGET_SIZE - cropped.shape[1]) // 2
            pad_y = (TARGET_SIZE - cropped.shape[0]) // 2
            padded = cv2.copyMakeBorder(cropped, pad_y, TARGET_SIZE - cropped.shape[0] - pad_y,
                                       pad_x, TARGET_SIZE - cropped.shape[1] - pad_x,
                                       cv2.BORDER_CONSTANT, value=0)
            
         # print(f'area: {padded.shape[0]*padded.shape[1]}')

         # cv2.imshow('tile padded', padded)
         # cv2.waitKey(0)
         # cv2.destroyAllWindows()

         blur = cv2.bilateralFilter(cropped, 5, 50, 50)  # smooth the tile

         tiles.append(padded)  # add to tiles

   return tiles

def _get_letter_from_tile(tile):
   transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5], std=[0.5])
   ])

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
         )
         self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pool  → (64,1,1)
         self.classifier = nn.Linear(128, n_classes)

      def forward(self, x):
         x = self.features(x)
         x = self.pool(x).flatten(1)
         return self.classifier(x)

   net = TinyCNN(26).to(DEVICE)
   net.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))

   with torch.no_grad():
      if len(tile.shape) == 3:
         tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

      # Transform to tensor
      tensor = transform(tile).unsqueeze(0).to(DEVICE)  # shape (1, 1, 90, 90)

      # Run inference
      logits = net(tensor)
      # pred_class = logits.argmax(dim=1).item()
      probs = torch.softmax(logits, dim=1)
      conf, pred_class = probs.max(dim=1)
      pred_letter = chr(ord('A') + pred_class.item())
      # print(f'LETTER: {pred_letter}, CONF: {conf.item():.3f}')

      return pred_letter


# extracts the board from the video and returns it as a matrix of characters
def extract_board(video_path, debug=True):
   frame = _get_frame(video_path, 160)  # starts at the 160th frame, around 2 seconds in. change later
   magic_pix = _magic_to_pixels(video_path, debug=debug)
   board_roi = _get_roi(frame, magic_pix, 'board', debug=debug)
   if debug:
      cv2.imshow('board roi raw', board_roi)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
   board_roi = _apply_preprocessing_board(board_roi, shape=frame.shape, debug=debug)
   if debug:
      cv2.imshow('board roi', board_roi)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
   
   tiles = _split_into_tiles(board_roi, 5)  # split into 5x5, (40,40) tiles
   string25 = ""
   for i, tile in enumerate(tiles):
      if debug:
         cv2.imshow(f'tile {i}', tile)
         cv2.waitKey(0)
         cv2.destroyAllWindows()
      letter = _get_letter_from_tile(tile)
      string25 += letter
      if debug:
         print(letter, end='', flush=True)
         if (i + 1) % 5 == 0:
            print(flush=True)
   

   # turn to a matrix
   # send = text.split('\n')
   # for i in range(len(send)):
   #    send[i] = list(send[i])
   
   return string25

def extract_board_from_image(image_path, debug=True):
   frame = cv2.imread(image_path)
   magic_pix = _magic_to_pixels_image(image_path, debug=debug)
   board_roi = _get_roi(frame, magic_pix, 'board', debug=debug)
   if debug:
      cv2.imshow('board roi raw', board_roi)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
   board_roi = _apply_preprocessing_board(board_roi, modality='image', shape=frame.shape, debug=debug)
   if debug:
      cv2.imshow('board roi', board_roi)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

   tiles = _split_into_tiles(board_roi, 5)  # split into 5x5, (90,90) tiles
   string25 = ""
   for i, tile in enumerate(tiles):
      letter = _get_letter_from_tile(tile)
      string25 += letter
      if debug:
         cv2.imshow(f'tile {i}', tile)
         cv2.waitKey(0)
         cv2.destroyAllWindows()
         print(letter, end='', flush=True)
         if (i + 1) % 5 == 0:
            print(flush=True)
   
   return string25

   

   
def _apply_preprocessing_word_count(roi):
   # make gray
   gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
   equalized = cv2.equalizeHist(gray)

   # adaptive threshold
   _, binary = cv2.threshold(equalized, 21, 255, cv2.THRESH_BINARY_INV)
   thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 8)

   # close letters to get rid of artifacts
   kernel_close = np.ones((1,1),np.uint8)
   closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

   # light erosion to thin bloated strokes
   eroded = cv2.erode(closing, np.ones((1,1), np.uint8), iterations=1)

   kernel_dilate = np.ones((2, 2), np.uint8)
   dilated = cv2.dilate(eroded, kernel_dilate, iterations=1)

   return dilated/255

def get_only_the_word(original_roi, vertical_band=(0.2, 0.75), dilate_kernel=(5, 5), iterations=3):
   h, w = original_roi.shape
   top = int(vertical_band[0] * h)
   bottom = int(vertical_band[1] * h)

   kern = np.ones(dilate_kernel, np.uint8)
   roi_fattened = cv2.dilate(original_roi[top:bottom], kern, iterations=iterations)

   # if pixels border on edge or corner, theyre in the same connected component
   num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_fattened, connectivity=8)

   # Find the blob that contains the center column
   center_col = w // 2
   selected_box = None
   for i in range(1, num_labels):  # skip background (0)
      x, y, bw, bh, area = stats[i]
      if x <= center_col <= x + bw:
         # Adjust y for cropping from the band
         selected_box = (x, y + top, bw, bh)
         break

   if selected_box is None:
      return original_roi  # fallback: no crop

   x, y, bw, bh = selected_box
   cropped = original_roi[y:y + bh, x:x + bw]
   return cropped


# extracts the words found in the video and returns them as a list of strings
def extract_words_found(video_path):
   # frame finding
   START_FRAME = 400
   END_FRAME = 12000#9000
   THRESHOLD = 0.018  # threshold for the difference BETWEEN FRAMES to consider a new word found
   # ocr and word detection
   VERTICAL_BAND = (0.2, 0.75)  # vertical band to consider for the word
   DILATE_KERNEL = (5, 5)  # kernel size for dilation
   DILATE_ITERATIONS = 3  # number of iterations for dilation
   
   cap = cv2.VideoCapture(video_path)
   cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)  # starts at the 160th frame, around 2 seconds in. change later
   magic_pix = _magic_to_pixels(video_path)

   new_words_frames = []
   all_metrics = []  # for debugging
   debug = _get_roi(cap.read()[1], magic_pix, 'word_count')
   prev_frame = _apply_preprocessing_word_count(debug)
   threshold = THRESHOLD*prev_frame.shape[0]*prev_frame.shape[1]

   print(f'number of pixels in word count ROI: {prev_frame.shape[0]*prev_frame.shape[1]}')
   print(f'shape of word count ROI: {prev_frame.shape}')
   print(debug)
   print("black and white")
   print(prev_frame)
   print("sum")
   print(f'{np.sum(prev_frame)} {np.max(prev_frame)}')
   current_frame = START_FRAME

   # while current_frame < END_FRAME:
   #    ret, frame = cap.read()
   #    current_frame += 1
   #    if not ret:
   #       print("Failed to read frame.")
   #       break

   #    word_count_roi = _get_roi(frame, magic_pix, 'word_count')
   #    word_count_roi = _apply_preprocessing_word_count(word_count_roi)
   #    metric = np.sum(np.abs(prev_frame-word_count_roi))
   #    print(f"Frame {current_frame}: metric = {metric}")

   #    prev_frame = word_count_roi
   #    if metric < threshold: 
   #       continue  # if the difference is too small, skip this frame
      
   #    all_metrics.append(metric)
   #    new_words_frames.append(current_frame)
   #    print(f"New word found at frame {current_frame}")
      # words_roi = _get_roi(frame, magic_pix, 'words')
      # cv2.imshow('words roi', words_roi)
      # cv2.imshow('word_count roi', word_count_roi)

      # key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press
      # if key == ord('q'):
      #    break
               
   new_words_frames = [496, 564, 600, 694, 864, 926, 1032, 1068, 1110, 1258, 1328, 1392, 1568, 2187, 2299, 2445, 2573, 2671, 2887, 3543, 3617, 3705, 3779, 4011, 4111, 4217, 4445, 5168, 5268, 5308, 5340, 5648, 5808, 5884, 5930, 6092, 6132, 6164, 6220, 6292, 6462, 6538, 6738, 7122, 7162, 7224, 7258, 7465, 7591, 7661, 7695, 8003, 8133, 8171, 8413, 8457, 8563, 8605, 8777, 9075, 9109, 9169, 9461, 9741, 9929, 9931]
   print(new_words_frames)
   print(all_metrics)
   print(sorted(all_metrics))

   for frame_num in new_words_frames:
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # go back 2 frames to get the word
      ret, frame = cap.read()
      if not ret:
         print(f"Failed to read frame {frame_num}.")
         continue

      roi = _get_roi(frame, magic_pix, 'words')
      roi = (_apply_preprocessing_word_count(roi)*255).astype(np.uint8)  # convert to uint8 for pytesseract
      # xdxd = [i for i in sorted(roi.flatten()) if i > 0 and i < 255]
      # print(f"Frame {frame_num} non-zero values: {len(xdxd)}")
      # print(xdxd)  # print non-zero values for debugging
      assert roi.dtype == np.uint8

      word = get_only_the_word(roi, vertical_band=VERTICAL_BAND, dilate_kernel=DILATE_KERNEL, iterations=DILATE_ITERATIONS)
      
      text = pytesseract.image_to_string(word, config=custom_config_words)
      # text = text.replace(" ", "").replace("|", "I").replace("\n\n", "\n").rstrip()
      print(f"Frame {frame_num}: {text.rstrip().lstrip()}")

      cv2.imshow(f'words roi', roi)
      cv2.imshow(f'words roi', word)
      print(f"Showing frame {frame_num}")

      key = cv2.waitKey(0) 
      cv2.destroyAllWindows()

      if key == ord('q'):
         break

   cv2.waitKey(0)
   cv2.destroyAllWindows()
   
def find_start_and_end(video_path):
   start_frame, end_frame = 0, 0
   return start_frame, end_frame

if __name__ == '__main__':
   # seek_thru_video(sample_video_120, start_frame=0)
   # transcode_to_120fps(sample_video, sample_video_120)
   # extract_words_found(sample_video_120)
   extract_board(r"C:\Users\neela\new_era\wordtrainer\backend\uploads\ScreenRecording_06-23-2025 19-51-01_1.mov")
   # extract_board_from_image(r"C:\Users\neela\new_era\wordtrainer\models_data\pics\IMG_6388.PNG")

   # videos = [os.path.abspath(os.path.join('./uploads',file)) for file in os.listdir('./uploads') if file.startswith('Screen')]
   # print(videos)
   # for video in videos[-1:]:
      # extract_board(video)

   #create
   # images = [os.path.abspath(os.path.join('../models_data/pics',file)) for file in os.listdir('../models_data/pics')]
   # with open("new_or_overwritten_file.txt", "w") as f:
   #    # print(images)
   #    for image in images:
   #       board = extract_board_from_image(image, debug=False)
   #       f.write(image)
   #       f.write("&-")
   #       f.write(board)
   #       f.write('\n')
   # videos = [os.path.abspath(os.path.join('./uploads',file)) for file in os.listdir('./uploads') if file.startswith('Screen')]
   # print(videos)
   # with open("new_or_overwritten_file.txt", "a") as f:
   #    for video in videos:
   #       board = extract_board(video, debug=False)
   #       f.write(video)
   #       f.write("&-")
   #       f.write(board)
   #       f.write('\n')


   #verify
   # with open("new_or_overwritten_file.txt", "r") as f:
   #    for line in f.readlines():
   #       path, answer = line.split("&-")
   #       answer = answer.strip()
   #       if not os.path.exists(path):
   #          continue
   #       if "ScreenRecording" in path:
   #          result = extract_board(path, debug=False)
   #       else:
   #          result = extract_board_from_image(path, debug=False)
   #       if answer == result:
   #          print(f"{path}: correct")
   #       else:
   #          print(f"{path}: NOT CORRECT. answer vs result below")
   #          print(f"----- {answer}")
   #          print(f"----- {result}")

   # absroot = os.path.abspath('../models_data/pics')
   # for impath in os.listdir(absroot):
   #    impath = os.path.join(absroot, impath)
   #    print(impath)
   #    extract_board_from_image(impath)

   

# cpp errors given by poor error handling in cv2. make sure file path is full.