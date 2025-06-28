import os, cv2

TRAIN_LOC = '../../models_data/train'

dir_path = '../../models_data/tiles_raw'
# for i in range(26):
   # letter = chr(ord('a') + i)
   # dir_path = f"{TRAIN_LOC}/{letter}"
   # now, for each tile in each letter directory, 255-tile it to flip from black on white to white on black
for tile in os.listdir(dir_path):
   tile_path = os.path.join(dir_path, tile)
   if tile.endswith('.png'):
      img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
      flipped_img = 255 - img
      flipped_tile_path = os.path.join(dir_path, tile)
      cv2.imwrite(flipped_tile_path, flipped_img)
      print(f"Flipped tile: {flipped_tile_path}")

## delete augmented tiles
# for i in range(26):
#    letter = chr(ord('A')+i)  # A-Z
#    letter_dir = f'{TRAIN_LOC}/{letter}'
#    for tile in os.listdir(letter_dir):  # returns local file name
#       tile_path = os.path.join(letter_dir, tile)
#       if tile.endswith('_augmented.png'):
#          os.remove(tile_path)
#          print(f"File '{tile_path}' deleted successfully.")

# for i in range(3):
#    letter = chr(ord('A')+i)  # A-Z
#    letter_dir = f'{TRAIN_LOC}/{letter}'
#    for tile in os.listdir(letter_dir):  # returns local file name
#       tile_path = os.path.join(letter_dir, tile)
#       image = cv2.imread(tile_path)
#       print(image.shape)  # 90x90x3