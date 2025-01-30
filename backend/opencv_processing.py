import cv2, pytesseract
import numpy as np

# VERY hardcoded for now. works for my phone (iPhone 13), might work for others
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
sample_video = './uploads/iphone12.mp4'

custom_config = '-l eng --psm 6' # -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"'

# need relatives. pixels never guaranteed to match
magic = {
   'time': {'ytop': 0.17604166666666668, 'ybottom': 0.203125, 'xtop': 0.6981981981981982, 'xbottom': 0.9234234234234234},
   'board': {'ytop': 0.42552083333333335, 'ybottom': 0.7927083333333333, 'xtop': 0.10247747747747747, 'xbottom': 0.8975225225225225},
   'words': {'ytop': 0.096875, 'ybottom': 0.11510416666666666, 'xtop': 0.006756756756756757, 'xbottom': 0.6295045045045045},
   'score': {'ytop': 0.11614583333333334, 'ybottom': 0.14895833333333333, 'xtop': 0.6216216216216216, 'xbottom': 0.8975225225225225}
}

def _magic_to_pixels(m, video_path):
   cap = cv2.VideoCapture(video_path)
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
   cap.release()
   n = {}
   for key in m.keys():
      n[key] = {k:(int(v*width) if k.startswith('x') else int(v*height)) for k,v in m[key].items()}
   return n

def _get_roi(frame, magic_pix, interest):
   interest_subdict = magic_pix[interest]
   roi = frame[interest_subdict['ytop']:interest_subdict['ybottom'], interest_subdict['xtop']:interest_subdict['xbottom']]
   return roi

def _get_frame(video_path, number):
   cap = cv2.VideoCapture(video_path)
   for _ in range(number):
      _, frame = cap.read()
   return frame

def _apply_preprocessing(roi):  # somehow this works 
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


def get_frame_with_start(video_path):
   cap = cv2.VideoCapture(video_path)
   fnum = 0
   print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
   fps = cap.get(cv2.CAP_PROP_FPS)
   print(f'fps is {fps}')
   magic_pix = _magic_to_pixels(magic, video_path)
   print(magic_pix)
   
   while True:
      ret, frame = cap.read()
      if not ret: break

      roi = _get_roi(frame, magic_pix, 'time')
      # cv2.imshow('roi', roi)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      # roi = _apply_preprocessing(roi)
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
   
def extract_board(video_path):
   frame = _get_frame(video_path, 160)
   magic_pix = _magic_to_pixels(magic, video_path)
   board_roi = _get_roi(frame, magic_pix, 'board')
   board_roi = _apply_preprocessing(board_roi)
   text = pytesseract.image_to_string(board_roi, config=custom_config)
   text = text.replace(" ", "").replace("|", "I").replace("\n\n", "\n").rstrip()
   print(text)
   # turn to a matrix
   send = text.split('\n')
   for i in range(len(send)):
      send[i] = list(send[i])
   # cv2.imshow('board roi', board_roi)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
   return send
   

def extract_words_found(video_path):
   frame = get_frame_with_start(video_path)
   magic_pix = _magic_to_pixels(magic, video_path)
   words_roi = _get_roi(frame, magic_pix, 'words')
   


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