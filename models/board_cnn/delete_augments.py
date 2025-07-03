import os, cv2
TRAIN_LOC = '../../models_data/train'
dir_path = '../../models_data/tiles_raw'

## delete augmented tiles
for i in range(26):
   letter = chr(ord('A')+i)  # A-Z
   letter_dir = f'{TRAIN_LOC}/{letter}'
   for tile in os.listdir(letter_dir):  # returns local file name
      tile_path = os.path.join(letter_dir, tile)
      if tile.endswith('_augmented.png'):
         os.remove(tile_path)
         print(f"File '{tile_path}' deleted successfully.")