import cv2, pytesseract
import numpy as np
import subprocess

# VERY hardcoded for now. works for my phone (iPhone 13), might work for others
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
sample_video = './uploads/iphone12.mp4'
sample_video_120 = './uploads/iphone12_120.mp4'

custom_config = '-l eng --psm 6' # -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"'
custom_config_words = '-l eng --psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+"'

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
def _magic_to_pixels(video_path):
   cap = cv2.VideoCapture(video_path)
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
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

# extracts the region of interest (ROI) from the frame based on the magic coordinates
# interest is a key in the magic dictionary: 'time', 'board', 'word_count', 'score'
def _get_roi(frame, magic_pix, interest):
   interest_subdict = magic_pix[interest]
   ytop = interest_subdict['ytop']
   ybottom = interest_subdict['ybottom']
   xleft = interest_subdict['xleft']
   xright = interest_subdict['xright']
   roi = frame[ytop:ybottom, xleft:xright]
   # roi = frame[interest_subdict['ytop']:interest_subdict['ybottom'], interest_subdict['xleft']:interest_subdict['xright']]
   return roi

# gets the nth frame from the video
def _get_frame(video_path, number):
   cap = cv2.VideoCapture(video_path)
   cap.set(cv2.CAP_PROP_POS_FRAMES, number)
   _, frame = cap.read()
   return frame

# applies preprocessing to the ROI to prepare it for OCR
def _apply_preprocessing_board(roi):  # somehow this works 
   # make gray
   gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
   equalized = cv2.equalizeHist(gray)

   # binary threshold: anything below 21 becomes white
   _, binary = cv2.threshold(equalized, 21, 255, cv2.THRESH_BINARY_INV)
   th2 = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

   # close letters to get rid of artifacts
   kernel = np.ones((5,5),np.uint8)
   closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

   # contours
   closing_contours = closing.copy()
   items = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   contours = items[0] if len(items) == 2 else items[1]
   cv2.drawContours(closing_contours, contours, -1, (0,0,255), 2)

   # want to isolate black letters
   lower = np.array([0,0,0])
   upper = np.array([180,100,100])
   # mask = cv2.inRange(hsv, lower, upper)

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


# gets the frame with the start time in it, and returns it
def get_frame_with_start(video_path):
   cap = cv2.VideoCapture(video_path)
   fnum = 0
   print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
   fps = cap.get(cv2.CAP_PROP_FPS)
   print(f'fps is {fps}')
   magic_pix = _magic_to_pixels(video_path)
   print(magic_pix)
   
   while True:
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
   
# extracts the board from the video and returns it as a matrix of characters
def extract_board(video_path):
   frame = _get_frame(video_path, 160)  # starts at the 160th frame, around 2 seconds in. change later
   magic_pix = _magic_to_pixels(video_path)
   board_roi = _get_roi(frame, magic_pix, 'board')
   board_roi = _apply_preprocessing_board(board_roi)
   text = pytesseract.image_to_string(board_roi, config=custom_config)
   text = text.replace(" ", "").replace("|", "I").replace("\n\n", "\n").rstrip()
   print(text)
   # turn to a matrix
   send = text.split('\n')
   for i in range(len(send)):
      send[i] = list(send[i])
   cv2.imshow('board roi', board_roi)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   return send
   
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
   # seek_thru_video(sample_video_120, start_frame=270)
   # transcode_to_120fps(sample_video, sample_video_120)
   extract_words_found(sample_video_120)
   # extract_board(sample_video)
   # frame = _get_frame(sample_video, 500)
   # frame = _get_frame(sample_video_120, 6132)
   # roi = _get_roi(frame, _magic_to_pixels(sample_video), 'words')
   # roi = (_apply_preprocessing_word_count(roi)*255).astype(np.uint8)  # convert to uint8 for pytesseract
   # cv2.imshow('words roi', roi)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()

# extract_board(sample_video)

# cpp errors given by poor error handling in cv2. make sure file path is full.
# frame = get_frame_with_start(sample_video)
# cap = cv2.VideoCapture(sample_video)
# fps = cap.get(cv2.CAP_PROP_FPS)

# i=0
# while True:
#    ret, frame = cap.read()
#    if not ret: break
#    i+=1
#    print(i, end=' ', flush=True)

#    if i==int(fps*2):
#       # convert magic to pixels
#       magic_pix = _magic_to_pixels(magic, sample_video)
#       print(magic_pix)
#       print(f'video resolution is {frame.shape[0]}x{frame.shape[1]}') 
#       cv2.imshow('Frame', _get_roi(frame, magic_pix, 'time'))
#       cv2.waitKey(0)
#       cv2.destroyAllWindows()
#       break

# get_frame_with_start(sample_video)

# img = cv2.imread('../iphone12ss.png', 0)
# print(f'shape of image is {img.shape[0]}x{img.shape[1]}')
# img = img[192:220, 375:460]
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('Image', frame) [375:460, 192:220]  y span, x span. goes from top->bottom, left->right

# board: [817:1522, 91:797]
# words: [186:221, 6:559]
# score: [223:286, 552:797]