import os, cv2

TRAIN_LOC = '../../models_data/train'

dir_path = '../../models_data/tiles_raw'
# for i in range(26):
   # letter = chr(ord('a') + i)
   # dir_path = f"{TRAIN_LOC}/{letter}"
   # now, for each tile in each letter directory, 255-tile it to flip from black on white to white on black
# for tile in os.listdir(dir_path):
#    tile_path = os.path.join(dir_path, tile)
#    if tile.endswith('.png'):
#       img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
#       flipped_img = 255 - img
#       flipped_tile_path = os.path.join(dir_path, tile)
#       cv2.imwrite(flipped_tile_path, flipped_img)
#       print(f"Flipped tile: {flipped_tile_path}")

def zoom_at(img, zoom=1, angle=0, coord=None): 
   cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
   
   rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
   result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
   
   return result

def zoom_at2(img, zoom=1.0):
   h, w = img.shape[:2]
   # Compute crop size
   new_h, new_w = int(h / zoom), int(w / zoom)
   y1 = (h - new_h) // 2
   x1 = (w - new_w) // 2
   cropped = img[y1:y1+new_h, x1:x1+new_w]
   return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

letter = 'W'
ws = os.listdir(f'../../models_data/train/{letter}')
imp = os.path.join(os.path.abspath(f'../../models_data/train/{letter}'), ws[0])
image = cv2.imread(imp)

cv2.imshow('reg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

zoom = zoom_at2(image, zoom=1.4)

cv2.imshow('zoom', zoom)
cv2.waitKey(0)
cv2.destroyAllWindows()